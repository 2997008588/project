# rag_app.py
# Streamlit: 课程智能答疑（PDF + FAISS + API LLM）
# - 多会话侧边栏（新聊天 + 历史对话可切换）
# - 默认多轮上下文 12 轮（隐藏，不在侧边栏显示）
# - 索引扫描 data/pdf（相对脚本目录），管理员可重建
# - 检索：向量召回 + 关键词重排 + 证据门槛（防止胡说）
# - 元数据用 JSON（避免 pickle 的 Chunk 类反序列化报错）

from __future__ import annotations

import os
import re
import json
import time
import uuid
import math
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import streamlit as st

# ---- Optional deps (fail gracefully) ----
try:
    import numpy as np
except Exception:
    np = None

try:
    import faiss  # faiss-cpu
except Exception:
    faiss = None

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    import requests
    import certifi
except Exception:
    requests = None
    certifi = None


# =========================
# Config
# =========================
APP_TITLE = "课程智能答疑（PDF + FAISS + API LLM）"

# 默认每个会话保留 12 轮问答（= 24 条 message）
HISTORY_TURNS = 12
MAX_MESSAGES = HISTORY_TURNS * 2

# 向量召回数量 + 最终用于生成的证据数量
VEC_RECALL_K = 40
EVIDENCE_TOP_K = 6

# 证据门槛：同时考虑“向量相似度”和“关键词覆盖”
MIN_VEC_SCORE = 0.30          # 余弦相似度（归一化 inner product）大概 0~1
MIN_KEYWORD_COVER = 0.08      # 关键词覆盖率（0~1），太低认为不相关

# Chunk 参数
CHUNK_SIZE = 900
CHUNK_OVERLAP = 160

# Embedding 模型（建议用小模型，部署快）
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# 目录：相对脚本所在目录（部署不受工作目录影响）
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdf"
INDEX_DIR = DATA_DIR / "index"
INDEX_FAISS_PATH = INDEX_DIR / "faiss.index"
INDEX_META_PATH = INDEX_DIR / "chunks.json"
MANIFEST_PATH = INDEX_DIR / "manifest.json"
CONV_STORE_PATH = INDEX_DIR / "conversations.json"


# =========================
# Helpers: safe secrets/env
# =========================
def safe_get_secret(key: str) -> str:
    """
    Streamlit 没有 secrets 文件时，访问 st.secrets 会抛 StreamlitSecretNotFoundError。
    这里统一捕获，没取到则返回空字符串。
    """
    try:
        # st.secrets 支持 dict-like 读取
        val = st.secrets.get(key, "")
        return str(val).strip() if val is not None else ""
    except Exception:
        return ""


def get_llm_cfg() -> Tuple[str, str, str]:
    """
    优先读取 st.secrets，其次读取环境变量。
    """
    base = safe_get_secret("LLM_BASE_URL") or os.getenv("LLM_BASE_URL", "").strip()
    model = safe_get_secret("LLM_MODEL") or os.getenv("LLM_MODEL", "").strip()
    key = safe_get_secret("LLM_API_KEY") or os.getenv("LLM_API_KEY", "").strip()
    # 默认值（Groq OpenAI compatible）
    if not base:
        base = "https://api.groq.com/openai/v1"
    if not model:
        model = "llama-3.1-8b-instant"
    return base, model, key


def get_admin_token() -> str:
    return safe_get_secret("ADMIN_TOKEN") or os.getenv("ADMIN_TOKEN", "").strip()


# =========================
# Text utils
# =========================
def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def text_hash(s: str) -> str:
    return hashlib.md5((s or "").encode("utf-8", errors="ignore")).hexdigest()


def tokenize(s: str) -> List[str]:
    """
    简易分词：英文/数字 token + 连续中文片段 token。
    不依赖额外库，够用来做“关键词覆盖”。
    """
    s = s.lower()
    toks = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]+", s)
    # 去掉很短的噪声 token（例如单个中文“的”等）
    out = []
    for t in toks:
        if len(t) == 1 and re.match(r"[\u4e00-\u9fff]", t):
            continue
        out.append(t)
    return out


def keyword_cover(query: str, text: str) -> float:
    """
    关键词覆盖率：query tokens 有多少比例出现在 text 中。
    """
    q = tokenize(query)
    if not q:
        return 0.0
    tset = set(tokenize(text))
    hit = sum(1 for x in q if x in tset)
    return hit / max(1, len(q))


def looks_like_noise_chunk(t: str) -> bool:
    """
    过滤“欢迎界面/路径选择界面/运行结果/截图”等低信息内容。
    这类 chunk 极易误召回，导致你看到的“乱七八糟”回答。
    """
    s = norm_ws(t)
    if len(s) < 60:
        return True

    # 常见 UI/截图噪声
    noise_phrases = [
        "欢迎界面", "路径选择界面", "运行结果", "界面如下", "如图", "截图",
        "单击", "点击", "按钮", "菜单", "安装", "配置", "选择", "下一步"
    ]
    # “图x-x” 并且很短，基本是图注
    if re.search(r"图\s*\d+\s*[-–]\s*\d+", s) and len(s) < 180:
        return True

    # 命中噪声短句且内容不长
    if any(p in s for p in noise_phrases) and len(s) < 220:
        return True

    return False


# =========================
# PDF -> chunks
# =========================
def scan_pdfs(pdf_dir: Path) -> List[Path]:
    if not pdf_dir.exists():
        return []
    pdfs = []
    for p in sorted(pdf_dir.iterdir()):
        if p.is_file() and p.suffix.lower() == ".pdf":
            pdfs.append(p)
    return pdfs


def read_pdf_pages(pdf_path: Path) -> List[Tuple[int, str]]:
    if PdfReader is None:
        raise RuntimeError("缺少 pypdf。请在 requirements.txt 安装 pypdf。")
    reader = PdfReader(str(pdf_path))
    pages: List[Tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        txt = norm_ws(txt)
        if txt:
            pages.append((i, txt))
    return pages


def chunk_pages(pages: List[Tuple[int, str]], source_file: str,
                chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Dict[str, Any]]:
    """
    将 (page_no, text) 切成 chunk，并记录 page 范围。
    """
    chunks: List[Dict[str, Any]] = []
    buf = ""
    start_page = None
    last_page = None

    def flush():
        nonlocal buf, start_page, last_page
        txt = norm_ws(buf)
        if txt and not looks_like_noise_chunk(txt):
            chunks.append({
                "chunk_id": f"{source_file}::p{start_page}-p{last_page}::{text_hash(txt)[:8]}",
                "file_name": source_file,
                "page_start": int(start_page or 1),
                "page_end": int(last_page or (start_page or 1)),
                "text": txt,
            })
        buf = ""
        start_page = None
        last_page = None

    for (pno, txt) in pages:
        if start_page is None:
            start_page = pno
        last_page = pno

        if not buf:
            buf = txt
        else:
            buf = buf + "\n" + txt

        # 超过 chunk_size 就切
        if len(buf) >= chunk_size:
            flush()
            # overlap：保留末尾一段作为下个 chunk 开头
            if overlap > 0:
                tail = txt[-overlap:] if len(txt) > overlap else txt
                buf = tail
                start_page = pno
                last_page = pno

    flush()
    return chunks


# =========================
# Embeddings / FAISS
# =========================
@st.cache_resource(show_spinner=False)
def load_embedder() -> Any:
    if SentenceTransformer is None:
        raise RuntimeError("缺少 sentence-transformers。请在 requirements.txt 安装 sentence-transformers。")
    model = SentenceTransformer(EMBED_MODEL_NAME)
    return model


def embed_texts(embedder: Any, texts: List[str]) -> Any:
    if np is None:
        raise RuntimeError("缺少 numpy。")
    vec = embedder.encode(
        texts,
        show_progress_bar=False,
        normalize_embeddings=True
    )
    return np.asarray(vec, dtype="float32")


def build_manifest(pdfs: List[Path], embed_model: str) -> Dict[str, Any]:
    items = []
    for p in pdfs:
        items.append({
            "name": p.name,
            "size": p.stat().st_size if p.exists() else 0,
            "mtime": int(p.stat().st_mtime) if p.exists() else 0,
        })
    return {
        "embed_model": embed_model,
        "pdfs": items,
    }


def manifest_changed(new_manifest: Dict[str, Any]) -> bool:
    if not MANIFEST_PATH.exists():
        return True
    try:
        old = json.loads(MANIFEST_PATH.read_text("utf-8"))
    except Exception:
        return True
    return old != new_manifest


def save_index(index: Any, chunks: List[Dict[str, Any]], manifest: Dict[str, Any]) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    if faiss is None:
        raise RuntimeError("缺少 faiss-cpu。请在 requirements.txt 安装 faiss-cpu。")
    faiss.write_index(index, str(INDEX_FAISS_PATH))
    INDEX_META_PATH.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), "utf-8")
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), "utf-8")


def load_index() -> Tuple[Optional[Any], List[Dict[str, Any]]]:
    if faiss is None:
        return None, []
    if not INDEX_FAISS_PATH.exists() or not INDEX_META_PATH.exists():
        return None, []
    try:
        idx = faiss.read_index(str(INDEX_FAISS_PATH))
        metas = json.loads(INDEX_META_PATH.read_text("utf-8"))
        if not isinstance(metas, list):
            metas = []
        return idx, metas
    except Exception:
        return None, []


def ensure_index_ready(embedder: Any, force_rebuild: bool = False) -> Tuple[Optional[Any], List[Dict[str, Any]], str]:
    pdfs = scan_pdfs(PDF_DIR)
    if not pdfs:
        return None, [], f"未发现 PDF：请将教材放到 {PDF_DIR.as_posix()} 下（方案 A）。"

    manifest = build_manifest(pdfs, EMBED_MODEL_NAME)
    need_rebuild = force_rebuild or manifest_changed(manifest) or (not INDEX_FAISS_PATH.exists()) or (not INDEX_META_PATH.exists())

    if not need_rebuild:
        idx, metas = load_index()
        if idx is not None and metas:
            return idx, metas, f"索引已就绪：chunks={len(metas)}，PDF={len(pdfs)}"
        need_rebuild = True

    # rebuild
    all_chunks: List[Dict[str, Any]] = []
    for p in pdfs:
        pages = read_pdf_pages(p)
        ch = chunk_pages(pages, source_file=p.name)
        all_chunks.extend(ch)

    if not all_chunks:
        return None, [], "已发现 PDF，但抽取不到可用文本（可能是扫描版图片 PDF，或文本抽取为空）。"

    texts = [c["text"] for c in all_chunks]
    vecs = embed_texts(embedder, texts)

    dim = vecs.shape[1]
    idx = faiss.IndexFlatIP(dim)
    idx.add(vecs)

    save_index(idx, all_chunks, manifest)
    return idx, all_chunks, f"索引构建完成：chunks={len(all_chunks)}，PDF={len(pdfs)}"


# =========================
# Retrieval (vector + keyword rerank)
# =========================
def search_chunks(index: Any, metas: List[Dict[str, Any]], embedder: Any, query: str,
                  recall_k: int = VEC_RECALL_K, top_k: int = EVIDENCE_TOP_K) -> List[Tuple[float, float, Dict[str, Any]]]:
    """
    return: List[(final_score, vec_score, meta)]
    """
    if np is None or faiss is None or index is None or not metas:
        return []

    qv = embed_texts(embedder, [query])
    D, I = index.search(qv, recall_k)

    candidates: List[Tuple[float, Dict[str, Any]]] = []
    for vec_score, idx in zip(D[0].tolist(), I[0].tolist()):
        if idx < 0 or idx >= len(metas):
            continue
        m = metas[idx]
        txt = m.get("text", "")
        if not txt:
            continue

        # 关键词覆盖（必须在重排里占一部分权重）
        kcov = keyword_cover(query, txt)

        # final score = 0.75*vec + 0.25*kcov（你也可以调）
        final = 0.75 * float(vec_score) + 0.25 * float(kcov)
        candidates.append((final, {
            **m,
            "_vec_score": float(vec_score),
            "_kcov": float(kcov),
        }))

    # 按 final 排序
    candidates.sort(key=lambda x: x[0], reverse=True)

    out: List[Tuple[float, float, Dict[str, Any]]] = []
    for final, m in candidates[:top_k]:
        out.append((final, float(m.get("_vec_score", 0.0)), m))
    return out


def evidence_is_enough(query: str, scored: List[Tuple[float, float, Dict[str, Any]]]) -> bool:
    """
    证据门槛：防止“命中欢迎界面/运行结果”这类内容时 LLM 开始胡写。
    规则：top1 同时满足向量分数 + 关键词覆盖；或 top3 中至少 2 条满足关键词覆盖。
    """
    if not scored:
        return False

    top1 = scored[0][2]
    v1 = float(top1.get("_vec_score", 0.0))
    k1 = float(top1.get("_kcov", 0.0))

    if v1 >= MIN_VEC_SCORE and k1 >= MIN_KEYWORD_COVER:
        return True

    good_kw = sum(1 for _, _, m in scored[:3] if float(m.get("_kcov", 0.0)) >= MIN_KEYWORD_COVER)
    return good_kw >= 2


def format_evidence(scored: List[Tuple[float, float, Dict[str, Any]]]) -> Tuple[str, List[Dict[str, Any]]]:
    """
    将 evidence 格式化成 LLM 可用的“编号资料块”，并返回用于展示的结构化列表。
    """
    blocks = []
    show_list: List[Dict[str, Any]] = []

    for i, (_, _, m) in enumerate(scored, start=1):
        fn = m.get("file_name", "文件.pdf")
        ps = m.get("page_start", "?")
        pe = m.get("page_end", "?")
        txt = m.get("text", "")

        blocks.append(f"[{i}] 来源：{fn} p{ps}-{pe}\n{txt}\n")

        show_list.append({
            "i": i,
            "file_name": fn,
            "page_start": ps,
            "page_end": pe,
            "vec": float(m.get("_vec_score", 0.0)),
            "kcov": float(m.get("_kcov", 0.0)),
            "text": txt,
        })

    return "\n".join(blocks).strip(), show_list


# =========================
# Multi-turn: build standalone query
# =========================
def needs_context(q: str) -> bool:
    q = q.strip()
    if len(q) <= 8:
        return True
    # 典型追问指代
    if any(x in q for x in ["它", "这", "有什么作用", "和它", "区别", "为什么", "怎么用"]):
        return True
    return False


def build_search_query(messages: List[Dict[str, str]], current_q: str) -> str:
    """
    用历史问题拼成更稳的检索 query（默认 12 轮，不在 UI 暴露）。
    """
    current_q = current_q.strip()
    if not messages:
        return current_q

    # 取历史里的 user 问题
    hist_q = [m["content"] for m in messages if m.get("role") == "user"]
    hist_q = [x.strip() for x in hist_q if x.strip()]

    if not hist_q:
        return current_q

    if needs_context(current_q):
        # 取最近 2 个问题做拼接更稳（避免过长）
        tail = hist_q[-2:] if len(hist_q) >= 2 else hist_q[-1:]
        return "；".join(tail + [current_q])

    return current_q


# =========================
# LLM call (OpenAI compatible)
# =========================
SYSTEM_PROMPT = """你是“课程智能答疑助手”。你只能依据我提供的“可用资料”回答问题。
强约束：
1) 只能使用资料中的信息，不允许编造、不允许推荐外部教材/网站。
2) 你的回答必须引用资料编号，例如：[1][3]。
3) 如果资料不足以回答，必须直接说“资料不足”，并说明你需要的关键词/章节方向，但仍然只能基于当前资料做推断。
4) 不要输出“根据提供的证据我无法找到”这类冗长废话；要么回答，要么资料不足。
"""

def call_llm(base: str, model: str, api_key: str, messages: List[Dict[str, str]],
             temperature: float = 0.2, max_tokens: int = 700) -> Tuple[Optional[str], Optional[str]]:
    """
    return (text, error)
    """
    if requests is None:
        return None, "缺少 requests 依赖。"
    if not api_key:
        return None, "未配置 LLM_API_KEY（请在 Streamlit Secrets 或环境变量设置）。"

    url = base.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": "Bearer " + api_key,
        "Content-Type": "application/json",
        "Connection": "close",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }

    s = requests.Session()
    s.trust_env = False

    last_err = None
    for _ in range(2):  # retry 2 times
        try:
            r = s.post(
                url,
                headers=headers,
                json=payload,
                timeout=45,
                verify=(certifi.where() if certifi else True),
            )
            if r.status_code >= 400:
                return None, f"HTTP {r.status_code}: {r.text[:200]}"
            data = r.json()
            text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
            return (text or "").strip(), None
        except Exception as e:
            last_err = str(e)
            time.sleep(0.8)

    return None, f"LLM 调用失败：{last_err}"


# =========================
# Conversation store
# =========================
def load_conv_store() -> Dict[str, Any]:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    if not CONV_STORE_PATH.exists():
        return {"active_id": "", "conversations": []}
    try:
        data = json.loads(CONV_STORE_PATH.read_text("utf-8"))
        if not isinstance(data, dict):
            return {"active_id": "", "conversations": []}
        if "conversations" not in data:
            data["conversations"] = []
        if "active_id" not in data:
            data["active_id"] = ""
        return data
    except Exception:
        return {"active_id": "", "conversations": []}


def save_conv_store(store: Dict[str, Any]) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    CONV_STORE_PATH.write_text(json.dumps(store, ensure_ascii=False, indent=2), "utf-8")


def unique_title(existing: List[str], base: str) -> str:
    base = base.strip() or "未命名对话"
    if base not in existing:
        return base
    k = 2
    while True:
        t = f"{base}({k})"
        if t not in existing:
            return t
        k += 1


def create_new_conversation(store: Dict[str, Any]) -> str:
    cid = uuid.uuid4().hex
    titles = [c.get("title", "") for c in store.get("conversations", [])]
    title = unique_title(titles, "未命名对话")
    store["conversations"].insert(0, {
        "id": cid,
        "title": title,
        "created_at": int(time.time()),
        "messages": []
    })
    store["active_id"] = cid
    save_conv_store(store)
    return cid


def get_active_conversation(store: Dict[str, Any]) -> Dict[str, Any]:
    active_id = store.get("active_id", "")
    for c in store.get("conversations", []):
        if c.get("id") == active_id:
            return c
    # 没有 active 就创建一个
    if store.get("conversations"):
        store["active_id"] = store["conversations"][0]["id"]
        save_conv_store(store)
        return store["conversations"][0]
    cid = create_new_conversation(store)
    return next(c for c in store["conversations"] if c["id"] == cid)


def set_conversation_title_if_needed(conv: Dict[str, Any], store: Dict[str, Any]) -> None:
    """
    用第一条 user 问题自动命名（不带 # 长串）。
    """
    if conv.get("title") and conv["title"] != "未命名对话":
        return
    msgs = conv.get("messages", [])
    first_q = ""
    for m in msgs:
        if m.get("role") == "user":
            first_q = m.get("content", "").strip()
            break
    if not first_q:
        return

    # 截断为 18 字左右
    name = first_q.replace("\n", " ")
    name = re.sub(r"\s+", " ", name).strip()
    if len(name) > 18:
        name = name[:18] + "…"

    titles = [c.get("title", "") for c in store.get("conversations", []) if c.get("id") != conv.get("id")]
    conv["title"] = unique_title(titles, name)
    save_conv_store(store)


def trim_messages(conv: Dict[str, Any]) -> None:
    msgs = conv.get("messages", [])
    if len(msgs) > MAX_MESSAGES:
        conv["messages"] = msgs[-MAX_MESSAGES:]


# =========================
# UI: Sidebar
# =========================
def sidebar_conversations(store: Dict[str, Any]) -> Dict[str, Any]:
    st.sidebar.header("对话")

    if st.sidebar.button("＋ 新聊天", use_container_width=True):
        create_new_conversation(store)

    convs = store.get("conversations", [])
    if not convs:
        create_new_conversation(store)
        convs = store.get("conversations", [])

    id_to_title = {c["id"]: c.get("title", "未命名对话") for c in convs}
    options = [c["id"] for c in convs]

    # 用 radio 显示展开列表（效果接近你截图那种“历史对话主题展开”）
    active_id = store.get("active_id", options[0])
    selected = st.sidebar.radio(
        label="",
        options=options,
        index=options.index(active_id) if active_id in options else 0,
        format_func=lambda cid: id_to_title.get(cid, "未命名对话"),
    )
    if selected != store.get("active_id", ""):
        store["active_id"] = selected
        save_conv_store(store)

    # 状态区
    with st.sidebar.expander("状态", expanded=False):
        base, model, key = get_llm_cfg()
        st.write("LLM_BASE_URL =", base)
        st.write("LLM_MODEL =", model)
        st.write("LLM_API_KEY =", ("已设置" if bool(key) else "未设置"))

    # Debug（可选）
    st.sidebar.checkbox("Debug", key="debug_mode")

    # 管理员区
    with st.sidebar.expander("管理员", expanded=False):
        admin_token = get_admin_token()
        if not admin_token:
            st.info("未设置 ADMIN_TOKEN（Secrets/环境变量）。将隐藏索引重建功能。")
            st.stop()

        entered = st.text_input("ADMIN_TOKEN", type="password", key="admin_token_input")
        ok = (entered.strip() == admin_token)
        if ok:
            st.success("管理员验证通过")
        else:
            st.warning("请输入正确 ADMIN_TOKEN")
            st.stop()

        if st.button("重建索引（扫描 data/pdf）", use_container_width=True):
            st.session_state.force_rebuild = True

    return store


# =========================
# Main app
# =========================
def render_chat(conv: Dict[str, Any]) -> None:
    for m in conv.get("messages", []):
        role = m.get("role", "assistant")
        content = m.get("content", "")
        with st.chat_message(role):
            st.markdown(content)


def answer_one_turn(conv: Dict[str, Any], store: Dict[str, Any],
                    index: Any, metas: List[Dict[str, Any]],
                    embedder: Any, user_q: str) -> None:
    # append user message
    conv.setdefault("messages", []).append({"role": "user", "content": user_q, "ts": int(time.time())})
    trim_messages(conv)
    save_conv_store(store)
    set_conversation_title_if_needed(conv, store)

    # build retrieval query (multi-turn)
    search_q = build_search_query(conv["messages"][:-1], user_q)

    # retrieve
    scored = search_chunks(index, metas, embedder, search_q)
    evidence_txt, evidence_list = format_evidence(scored)

    # decide enough?
    enough = evidence_is_enough(search_q, scored)

    base, model, key = get_llm_cfg()

    with st.chat_message("assistant"):
        if not enough:
            st.markdown("资料不足：当前 PDF 片段与问题不匹配，无法基于现有资料给出可靠回答。")
            if st.session_state.get("debug_mode"):
                st.markdown(f"**本轮用于检索的问题：** {search_q}")
            # 展示 topK 证据，方便你判断是否该调 chunk/模型
            with st.expander("本次检索到的证据（Top-K）", expanded=True):
                for e in evidence_list:
                    st.markdown(f"**[{e['i']}] {e['file_name']} p{e['page_start']}-{e['page_end']}**  "
                                f"(vec={e['vec']:.3f}, kcov={e['kcov']:.3f})")
                    st.write(e["text"][:1200])
            answer = "资料不足：检索到的片段与问题关键词不匹配，无法基于现有PDF给出可靠回答。"
        else:
            # Build messages: system + (history last N) + user with evidence
            hist = conv.get("messages", [])[:-1]
            # 只取最近 MAX_MESSAGES 以免上下文过长
            hist = hist[-MAX_MESSAGES:]

            llm_messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
            # 把历史对话带上（多轮对话）
            for m in hist:
                if m.get("role") in ("user", "assistant"):
                    llm_messages.append({"role": m["role"], "content": m.get("content", "")})

            user_payload = f"""问题：{user_q}

可用资料（只能使用这些资料回答，且必须引用编号）：
{evidence_txt}

请直接作答（中文，条理清晰，引用格式如：[1][3]）。"""
            llm_messages.append({"role": "user", "content": user_payload})

            if st.session_state.get("debug_mode"):
                st.markdown(f"**本轮用于检索的问题：** {search_q}")

            with st.spinner("正在生成回答..."):
                text, err = call_llm(base, model, key, llm_messages)

            if err or not text:
                st.error(err or "LLM 返回空内容")
                # 失败时退回“资料摘要”
                with st.expander("已返回检索到的资料摘要（未生成扩展解释）", expanded=True):
                    for e in evidence_list:
                        st.markdown(f"**[{e['i']}] {e['file_name']} p{e['page_start']}-{e['page_end']}**")
                        st.write(e["text"][:1200])
                answer = f"LLM 调用失败：{err}"
            else:
                st.markdown(text)
                # 展示证据
                with st.expander("本次检索到的证据（Top-K）", expanded=False):
                    for e in evidence_list:
                        st.markdown(f"**[{e['i']}] {e['file_name']} p{e['page_start']}-{e['page_end']}**  "
                                    f"(vec={e['vec']:.3f}, kcov={e['kcov']:.3f})")
                        st.write(e["text"][:1200])
                answer = text

    # append assistant
    conv["messages"].append({"role": "assistant", "content": answer, "ts": int(time.time())})
    trim_messages(conv)
    save_conv_store(store)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    # init store
    store = load_conv_store()
    store = sidebar_conversations(store)
    conv = get_active_conversation(store)

    # embedder
    try:
        embedder = load_embedder()
    except Exception as e:
        st.error(f"Embedding 初始化失败：{e}")
        st.stop()

    # index ready
    force = bool(st.session_state.get("force_rebuild", False))
    try:
        idx, metas, status = ensure_index_ready(embedder, force_rebuild=force)
        st.session_state.force_rebuild = False
    except Exception as e:
        st.error(f"索引初始化失败：{e}")
        st.stop()

    st.info(status)

    if idx is None or not metas:
        st.stop()

    # chat render
    render_chat(conv)

    user_q = st.chat_input("请输入你的问题（支持多轮追问）")
    if user_q:
        answer_one_turn(conv, store, idx, metas, embedder, user_q)


if __name__ == "__main__":
    main()
