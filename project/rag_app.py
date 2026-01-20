# rag_app.py
# Streamlit: 课程智能答疑（PDF + FAISS + API LLM）
# - 多会话侧边栏（新聊天 + 历史对话可切换）
# - 默认多轮上下文 12 轮（隐藏，不在 UI 暴露）
# - 索引扫描 data/pdf（相对脚本目录），管理员可重建
# - 检索：向量召回 + 关键词重排 + 证据门槛（防止胡说）
# - 元数据用 JSON（避免 pickle 反序列化报错）
# - conversations.json 自动迁移/清洗（修复你现在的 TypeError）

from __future__ import annotations

import os
import re
import json
import time
import uuid
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

# Embedding 模型
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
    s = (s or "").lower()
    toks = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]+", s)
    out = []
    for t in toks:
        if len(t) == 1 and re.match(r"[\u4e00-\u9fff]", t):
            continue
        out.append(t)
    return out


def keyword_cover(query: str, text: str) -> float:
    q = tokenize(query)
    if not q:
        return 0.0
    tset = set(tokenize(text))
    hit = sum(1 for x in q if x in tset)
    return hit / max(1, len(q))


def looks_like_noise_chunk(t: str) -> bool:
    """
    过滤“欢迎界面/路径选择界面/运行结果/截图”等低信息内容。
    """
    s = norm_ws(t)
    if len(s) < 60:
        return True

    noise_phrases = [
        "欢迎界面", "路径选择界面", "运行结果", "界面如下", "如图", "截图",
        "单击", "点击", "按钮", "菜单", "安装", "配置", "选择", "下一步"
    ]
    if re.search(r"图\s*\d+\s*[-–]\s*\d+", s) and len(s) < 180:
        return True
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


def chunk_pages(
    pages: List[Tuple[int, str]],
    source_file: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> List[Dict[str, Any]]:
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

        if len(buf) >= chunk_size:
            flush()
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
    return SentenceTransformer(EMBED_MODEL_NAME)


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
    return {"embed_model": embed_model, "pdfs": items}


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
    if np is None or faiss is None or index is None or not metas:
        return []

    qv = embed_texts(embedder, [query])
    D, I = index.search(qv, recall_k)

    candidates: List[Tuple[float, Dict[str, Any]]] = []
    for vec_score, idx_i in zip(D[0].tolist(), I[0].tolist()):
        if idx_i < 0 or idx_i >= len(metas):
            continue
        m = metas[idx_i]
        txt = m.get("text", "")
        if not txt:
            continue

        kcov = keyword_cover(query, txt)
        final = 0.75 * float(vec_score) + 0.25 * float(kcov)

        mm = dict(m)
        mm["_vec_score"] = float(vec_score)
        mm["_kcov"] = float(kcov)
        candidates.append((final, mm))

    candidates.sort(key=lambda x: x[0], reverse=True)

    out: List[Tuple[float, float, Dict[str, Any]]] = []
    for final, m in candidates[:top_k]:
        out.append((final, float(m.get("_vec_score", 0.0)), m))
    return out


def evidence_is_enough(query: str, scored: List[Tuple[float, float, Dict[str, Any]]]) -> bool:
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
    q = (q or "").strip()
    if len(q) <= 8:
        return True
    if any(x in q for x in ["它", "这", "有什么作用", "和它", "区别", "为什么", "怎么用"]):
        return True
    return False


def build_search_query(messages: List[Dict[str, str]], current_q: str) -> str:
    current_q = (current_q or "").strip()
    if not messages:
        return current_q

    hist_q = [m.get("content", "") for m in messages if m.get("role") == "user"]
    hist_q = [x.strip() for x in hist_q if x.strip()]
    if not hist_q:
        return current_q

    if needs_context(current_q):
        tail = hist_q[-2:] if len(hist_q) >= 2 else hist_q[-1:]
        return "；".join(tail + [current_q])

    return current_q


# =========================
# LLM call (OpenAI compatible)
# =========================
SYSTEM_PROMPT = """你是“课程智能答疑助手”。你只能依据我提供的“可用资料”回答问题。
强约束：
1) 只能使用资料中的信息，不允许编造、不允许推荐外部教材/网站。
2) 回答必须引用资料编号，例如：[1][3]。
3) 如果资料不足以回答，必须直接说“资料不足”，并说明你需要的关键词/章节方向（不要写废话）。
"""

def call_llm(base: str, model: str, api_key: str, messages: List[Dict[str, str]],
             temperature: float = 0.2, max_tokens: int = 700) -> Tuple[Optional[str], Optional[str]]:
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
    for _ in range(2):
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
# Conversation store (FIXED)
# =========================
def _clean_messages(msgs: Any) -> List[Dict[str, Any]]:
    if not isinstance(msgs, list):
        return []
    out: List[Dict[str, Any]] = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        if role not in ("user", "assistant", "system"):
            continue
        if content is None:
            continue
        out.append({
            "role": role,
            "content": str(content),
            "ts": int(m.get("ts") or time.time())
        })
    return out


def normalize_conv_store(data: Any) -> Dict[str, Any]:
    """
    兼容旧版本/错误格式，统一清洗成：
    {
      "active_id": str,
      "conversations": [
        {"id": str, "title": str, "created_at": int, "messages": [...]},
        ...
      ]
    }
    """
    if not isinstance(data, dict):
        data = {}

    convs = data.get("conversations", [])
    # 兼容 conversations 是 dict（按 id 做 key）
    if isinstance(convs, dict):
        convs = list(convs.values())
    if not isinstance(convs, list):
        convs = []

    cleaned: List[Dict[str, Any]] = []
    for c in convs:
        if not isinstance(c, dict):
            continue

        cid = c.get("id") or c.get("cid") or c.get("conversation_id")
        if not isinstance(cid, str) or not cid.strip():
            cid = uuid.uuid4().hex

        title = c.get("title") or c.get("name") or "未命名对话"
        if not isinstance(title, str) or not title.strip():
            title = "未命名对话"

        created_at = c.get("created_at")
        try:
            created_at = int(created_at) if created_at is not None else int(time.time())
        except Exception:
            created_at = int(time.time())

        msgs = c.get("messages") or c.get("history") or []
        msgs = _clean_messages(msgs)

        cleaned.append({
            "id": cid,
            "title": title,
            "created_at": created_at,
            "messages": msgs
        })

    active_id = data.get("active_id", "")
    if not isinstance(active_id, str):
        active_id = ""

    # 如果 active_id 不存在或不匹配，指向第一条
    if cleaned:
        ids = {c["id"] for c in cleaned}
        if active_id not in ids:
            active_id = cleaned[0]["id"]
    else:
        active_id = ""

    return {"active_id": active_id, "conversations": cleaned}


def load_conv_store() -> Dict[str, Any]:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    if not CONV_STORE_PATH.exists():
        store = {"active_id": "", "conversations": []}
        return store

    try:
        raw = json.loads(CONV_STORE_PATH.read_text("utf-8"))
    except Exception:
        # 文件坏了就直接重置
        return {"active_id": "", "conversations": []}

    store = normalize_conv_store(raw)

    # 如果清洗后结构有变化，直接回写，避免下次再炸
    try:
        CONV_STORE_PATH.write_text(json.dumps(store, ensure_ascii=False, indent=2), "utf-8")
    except Exception:
        pass

    return store


def save_conv_store(store: Dict[str, Any]) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    store = normalize_conv_store(store)
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
    store = normalize_conv_store(store)
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
    store = normalize_conv_store(store)
    active_id = store.get("active_id", "")
    for c in store.get("conversations", []):
        if c.get("id") == active_id:
            return c

    if store.get("conversations"):
        store["active_id"] = store["conversations"][0]["id"]
        save_conv_store(store)
        return store["conversations"][0]

    cid = create_new_conversation(store)
    store = load_conv_store()
    for c in store.get("conversations", []):
        if c.get("id") == cid:
            return c
    # fallback
    return {"id": cid, "title": "未命名对话", "created_at": int(time.time()), "messages": []}


def set_conversation_title_if_needed(conv: Dict[str, Any], store: Dict[str, Any]) -> None:
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

    name = re.sub(r"\s+", " ", first_q.replace("\n", " ")).strip()
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
    store = normalize_conv_store(store)

    st.sidebar.header("对话")
    if st.sidebar.button("＋ 新聊天", use_container_width=True):
        create_new_conversation(store)
        store = load_conv_store()

    convs = store.get("conversations", [])
    if not convs:
        create_new_conversation(store)
        store = load_conv_store()
        convs = store.get("conversations", [])

    # 防御式构造（避免 c 不是 dict 再炸）
    options: List[str] = []
    id_to_title: Dict[str, str] = {}
    for c in convs:
        if not isinstance(c, dict):
            continue
        cid = c.get("id")
        if not isinstance(cid, str) or not cid:
            continue
        options.append(cid)
        title = c.get("title", "未命名对话")
        id_to_title[cid] = title if isinstance(title, str) and title.strip() else "未命名对话"

    if not options:
        # 极端情况：清洗后仍空，重建一个
        create_new_conversation(store)
        store = load_conv_store()
        convs = store.get("conversations", [])
        options = [convs[0]["id"]] if convs else []
        id_to_title = {convs[0]["id"]: convs[0].get("title", "未命名对话")} if convs else {}

    active_id = store.get("active_id", options[0] if options else "")
    if active_id not in options and options:
        active_id = options[0]
        store["active_id"] = active_id
        save_conv_store(store)

    selected = st.sidebar.radio(
        label="",
        options=options,
        index=options.index(active_id) if active_id in options else 0,
        format_func=lambda cid: id_to_title.get(cid, "未命名对话"),
    )
    if selected != store.get("active_id", ""):
        store["active_id"] = selected
        save_conv_store(store)

    with st.sidebar.expander("状态", expanded=False):
        base, model, key = get_llm_cfg()
        st.write("LLM_BASE_URL =", base)
        st.write("LLM_MODEL =", model)
        st.write("LLM_API_KEY =", ("已设置" if bool(key) else "未设置"))

    st.sidebar.checkbox("Debug", key="debug_mode")

    # 管理员区：不再 st.stop() 阻塞应用，只做“隐藏/显示”控制
    with st.sidebar.expander("管理员", expanded=False):
        admin_token = get_admin_token()
        if not admin_token:
            st.caption("未设置 ADMIN_TOKEN（Secrets/环境变量），将隐藏“重建索引”。")
        else:
            entered = st.text_input("ADMIN_TOKEN", type="password", key="admin_token_input")
            ok = (entered.strip() == admin_token)
            if ok:
                st.success("管理员验证通过")
                if st.button("重建索引（扫描 data/pdf）", use_container_width=True):
                    st.session_state.force_rebuild = True
            else:
                st.caption("输入正确 ADMIN_TOKEN 才显示重建按钮。")

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


def answer_one_turn(
    conv: Dict[str, Any],
    store: Dict[str, Any],
    index: Any,
    metas: List[Dict[str, Any]],
    embedder: Any,
    user_q: str
) -> None:
    conv.setdefault("messages", []).append({"role": "user", "content": user_q, "ts": int(time.time())})
    trim_messages(conv)
    save_conv_store(store)
    set_conversation_title_if_needed(conv, store)

    search_q = build_search_query(conv["messages"][:-1], user_q)
    scored = search_chunks(index, metas, embedder, search_q)
    evidence_txt, evidence_list = format_evidence(scored)
    enough = evidence_is_enough(search_q, scored)

    base, model, key = get_llm_cfg()

    with st.chat_message("assistant"):
        if not enough:
            st.markdown("资料不足：当前 PDF 片段与问题不匹配，无法基于现有资料给出可靠回答。")
            if st.session_state.get("debug_mode"):
                st.markdown(f"**本轮用于检索的问题：** {search_q}")
            with st.expander("本次检索到的证据（Top-K）", expanded=True):
                for e in evidence_list:
                    st.markdown(
                        f"**[{e['i']}] {e['file_name']} p{e['page_start']}-{e['page_end']}**  "
                        f"(vec={e['vec']:.3f}, kcov={e['kcov']:.3f})"
                    )
                    st.write(e["text"][:1200])
            answer = "资料不足：检索到的片段与问题关键词不匹配，无法基于现有PDF给出可靠回答。"
        else:
            hist = conv.get("messages", [])[:-1]
            hist = hist[-MAX_MESSAGES:]

            llm_messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
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
                with st.expander("已返回检索到的资料摘要（未生成扩展解释）", expanded=True):
                    for e in evidence_list:
                        st.markdown(f"**[{e['i']}] {e['file_name']} p{e['page_start']}-{e['page_end']}**")
                        st.write(e["text"][:1200])
                answer = f"LLM 调用失败：{err}"
            else:
                st.markdown(text)
                with st.expander("本次检索到的证据（Top-K）", expanded=False):
                    for e in evidence_list:
                        st.markdown(
                            f"**[{e['i']}] {e['file_name']} p{e['page_start']}-{e['page_end']}**  "
                            f"(vec={e['vec']:.3f}, kcov={e['kcov']:.3f})"
                        )
                        st.write(e["text"][:1200])
                answer = text

    conv["messages"].append({"role": "assistant", "content": answer, "ts": int(time.time())})
    trim_messages(conv)
    save_conv_store(store)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    store = load_conv_store()
    store = sidebar_conversations(store)
    store = load_conv_store()  # sidebar 操作后再读一次确保一致
    conv = get_active_conversation(store)

    try:
        embedder = load_embedder()
    except Exception as e:
        st.error(f"Embedding 初始化失败：{e}")
        st.stop()

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

    render_chat(conv)

    user_q = st.chat_input("请输入你的问题（支持多轮追问）")
    if user_q:
        answer_one_turn(conv, store, idx, metas, embedder, user_q)


if __name__ == "__main__":
    main()
