# rag_app.py
# Streamlit RAG App: PDF + FAISS + API LLM (OpenAI-compatible)
# Features:
# - Sidebar conversation list + "New Chat" button
# - Default 12 turns per conversation (auto trim + optional summary)
# - Multi-turn coherence via question rewriting + dual retrieval (raw + rewritten)
# - Multi-PDF indexing from data/pdf/
# - Persistent index via faiss + JSONL (no pickle custom class)
# - Safe secrets access (no StreamlitSecretNotFoundError when secrets missing)

from __future__ import annotations

import os
import re
import json
import time
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import certifi
import numpy as np
import requests
import streamlit as st

try:
    import faiss  # faiss-cpu
except Exception as e:
    faiss = None

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


# ----------------------------
# Config
# ----------------------------
APP_TITLE = "课程智能答疑（PDF + FAISS + API LLM）"

PDF_DIR = Path("data/pdf")
INDEX_DIR = Path("data/index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

INDEX_FAISS_PATH = INDEX_DIR / "faiss.index"
INDEX_META_PATH = INDEX_DIR / "chunks.jsonl"
INDEX_MANIFEST_PATH = INDEX_DIR / "manifest.json"

# Chunking
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

# Retrieval
TOPK = 6
MIN_SCORE = 0.18  # cosine similarity threshold (IndexFlatIP)

# Conversation
MAX_TURNS_DEFAULT = 12  # 12 rounds of Q/A -> keep last 24 messages
AUTO_SUMMARIZE = True  # summarize older messages when trimming

# Embeddings
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# LLM env/secrets keys (OpenAI-compatible endpoint)
ENV_BASE = "LLM_BASE_URL"
ENV_MODEL = "LLM_MODEL"
ENV_KEY = "LLM_API_KEY"

# Admin
ENV_ADMIN = "ADMIN_TOKEN"

FOLLOWUP_TRIGGERS = (
    "它", "这", "那", "有什么", "怎么", "为什么", "区别", "关系", "作用", "如何", "能不能", "是不是", "这个", "那个"
)


# ----------------------------
# Utilities
# ----------------------------
def now_ms() -> int:
    return int(time.time() * 1000)


def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def safe_secrets_get(key: str) -> Optional[str]:
    """
    Streamlit will raise StreamlitSecretNotFoundError if secrets.toml does not exist
    and you touch st.secrets. So we must guard with try/except.
    """
    try:
        return st.secrets.get(key)  # type: ignore[attr-defined]
    except Exception:
        return None


def get_env_or_secret(key: str, default: str = "") -> str:
    v = safe_secrets_get(key)
    if v is None or str(v).strip() == "":
        v = os.getenv(key, default)
    return (str(v) if v is not None else default).strip()


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms


def short_title_from_text(text: str, max_len: int = 18) -> str:
    t = re.sub(r"\s+", " ", (text or "").strip())
    if not t:
        return "未命名对话"
    return (t[:max_len] + "…") if len(t) > max_len else t


# ----------------------------
# LLM Client (OpenAI-compatible)
# ----------------------------
class LLMError(Exception):
    pass


def llm_chat(messages: List[Dict[str, str]], base_url: str, model: str, api_key: str, timeout: int = 40) -> str:
    if not base_url or not model or not api_key:
        raise LLMError("LLM 配置不完整：请设置 LLM_BASE_URL / LLM_MODEL / LLM_API_KEY（secrets 或环境变量）。")

    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
    }

    s = requests.Session()
    s.trust_env = False  # avoid proxy issues
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Connection": "close",
    }

    try:
        r = s.post(url, headers=headers, json=payload, timeout=timeout, verify=certifi.where())
    except requests.exceptions.SSLError as e:
        raise LLMError(f"SSLError: {e}")
    except Exception as e:
        raise LLMError(f"请求失败: {type(e).__name__}: {e}")

    if r.status_code == 401:
        raise LLMError("HTTP 401：未授权。请检查 LLM_API_KEY 是否正确。")
    if r.status_code == 403:
        raise LLMError(f"HTTP 403：Forbidden。可能是 Key 权限/配额/来源限制。返回：{r.text[:200]}")
    if r.status_code >= 400:
        raise LLMError(f"HTTP {r.status_code}：{r.text[:200]}")

    try:
        data = r.json()
        return (data["choices"][0]["message"]["content"] or "").strip()
    except Exception:
        raise LLMError(f"响应解析失败：{r.text[:200]}")


# ----------------------------
# Embedding Model
# ----------------------------
@st.cache_resource(show_spinner=False)
def load_embedder(model_name: str) -> Any:
    if SentenceTransformer is None:
        raise RuntimeError("缺少 sentence-transformers。请在 requirements.txt 安装 sentence-transformers。")
    return SentenceTransformer(model_name)


def embed_texts(embedder: Any, texts: List[str]) -> np.ndarray:
    vecs = embedder.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    return np.asarray(vecs, dtype=np.float32)


def embed_query(embedder: Any, text: str) -> np.ndarray:
    v = embedder.encode([text], show_progress_bar=False, normalize_embeddings=True)
    return np.asarray(v, dtype=np.float32)


# ----------------------------
# PDF -> Chunks
# ----------------------------
@dataclass
class ChunkMeta:
    chunk_id: int
    source_file: str
    page_start: int
    page_end: int
    text: str


def read_pdf_text(pdf_path: Path) -> List[str]:
    if PdfReader is None:
        raise RuntimeError("缺少 pypdf。请在 requirements.txt 安装 pypdf。")
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, p in enumerate(reader.pages):
        try:
            t = p.extract_text() or ""
        except Exception:
            t = ""
        pages.append(t)
    return pages


def chunk_pages(pages: List[str], source_file: str, chunk_size: int, overlap: int) -> List[ChunkMeta]:
    all_chunks: List[ChunkMeta] = []
    chunk_id = 0

    def clean(s: str) -> str:
        s = s.replace("\u00a0", " ")
        s = re.sub(r"[ \t]+", " ", s)
        s = re.sub(r"\n{3,}", "\n\n", s)
        return s.strip()

    for page_idx, page_text in enumerate(pages, start=1):
        t = clean(page_text)
        if not t:
            continue

        # simple character window chunking
        start = 0
        while start < len(t):
            end = min(len(t), start + chunk_size)
            piece = t[start:end].strip()
            if piece:
                all_chunks.append(
                    ChunkMeta(
                        chunk_id=chunk_id,
                        source_file=source_file,
                        page_start=page_idx,
                        page_end=page_idx,
                        text=piece,
                    )
                )
                chunk_id += 1
            if end >= len(t):
                break
            start = max(0, end - overlap)

    return all_chunks


def scan_pdfs(pdf_dir: Path) -> List[Path]:
    if not pdf_dir.exists():
        return []
    pdfs = sorted([p for p in pdf_dir.glob("*.pdf") if p.is_file()])
    return pdfs


def build_manifest(pdfs: List[Path], embed_model_name: str) -> Dict[str, Any]:
    files = []
    for p in pdfs:
        st_ = p.stat()
        files.append({"name": p.name, "size": st_.st_size, "mtime": int(st_.st_mtime)})
    return {
        "embed_model": embed_model_name,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "files": files,
    }


def manifest_changed(new_manifest: Dict[str, Any]) -> bool:
    if not INDEX_MANIFEST_PATH.exists():
        return True
    try:
        old = json.loads(INDEX_MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        return True
    return old != new_manifest


def save_manifest(m: Dict[str, Any]) -> None:
    INDEX_MANIFEST_PATH.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")


def save_chunks_jsonl(chunks: List[ChunkMeta]) -> None:
    with INDEX_META_PATH.open("w", encoding="utf-8") as f:
        for c in chunks:
            obj = {
                "chunk_id": c.chunk_id,
                "source_file": c.source_file,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "text": c.text,
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_chunks_jsonl() -> List[ChunkMeta]:
    chunks: List[ChunkMeta] = []
    if not INDEX_META_PATH.exists():
        return chunks
    with INDEX_META_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            chunks.append(
                ChunkMeta(
                    chunk_id=int(obj["chunk_id"]),
                    source_file=str(obj["source_file"]),
                    page_start=int(obj["page_start"]),
                    page_end=int(obj["page_end"]),
                    text=str(obj["text"]),
                )
            )
    return chunks


def build_faiss_index(vectors: np.ndarray) -> Any:
    if faiss is None:
        raise RuntimeError("缺少 faiss-cpu。请在 requirements.txt 安装 faiss-cpu。")
    d = vectors.shape[1]
    index = faiss.IndexFlatIP(d)  # cosine similarity if vectors normalized
    index.add(vectors)
    return index


def save_faiss_index(index: Any) -> None:
    if faiss is None:
        raise RuntimeError("faiss 不可用。")
    faiss.write_index(index, str(INDEX_FAISS_PATH))


def load_faiss_index() -> Optional[Any]:
    if faiss is None:
        return None
    if not INDEX_FAISS_PATH.exists():
        return None
    try:
        return faiss.read_index(str(INDEX_FAISS_PATH))
    except Exception:
        return None


def ensure_index_ready(embedder: Any, force_rebuild: bool = False) -> Tuple[Optional[Any], List[ChunkMeta], str]:
    """
    Returns: (faiss_index, chunks, status_text)
    """
    pdfs = scan_pdfs(PDF_DIR)
    if not pdfs:
        return None, [], f"未发现 PDF：请将教材放到 {PDF_DIR.as_posix()}/ 下（方案 A）。"

    manifest = build_manifest(pdfs, EMBED_MODEL_NAME)

    need_rebuild = force_rebuild or manifest_changed(manifest) or (not INDEX_FAISS_PATH.exists()) or (not INDEX_META_PATH.exists())

    if not need_rebuild:
        idx = load_faiss_index()
        metas = load_chunks_jsonl()
        if idx is not None and metas:
            return idx, metas, f"索引已就绪（PDF {len(pdfs)} 个，chunks {len(metas)}）。"
        need_rebuild = True

    # Rebuild
    all_chunks: List[ChunkMeta] = []
    for p in pdfs:
        pages = read_pdf_text(p)
        chunks = chunk_pages(pages, source_file=p.name, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        all_chunks.extend(chunks)

    if not all_chunks:
        return None, [], "PDF 可读取但未提取到文本（可能是扫描版/图片 PDF）。"

    texts = [c.text for c in all_chunks]
    vecs = embed_texts(embedder, texts)  # already normalized embeddings
    idx = build_faiss_index(vecs)

    save_faiss_index(idx)
    save_chunks_jsonl(all_chunks)
    save_manifest(manifest)

    return idx, all_chunks, f"索引已重建（PDF {len(pdfs)} 个，chunks {len(all_chunks)}）。"


# ----------------------------
# Retrieval + Multi-turn Coherence
# ----------------------------
def is_followup(q: str) -> bool:
    q = (q or "").strip()
    if len(q) <= 12:
        return True
    return any(t in q for t in FOLLOWUP_TRIGGERS)


def conv_context_text(conv: Dict[str, Any], last_k_turns: int = 4) -> str:
    msgs = conv.get("messages", [])
    tail = msgs[-last_k_turns * 2 :]
    lines = []
    for m in tail:
        role = m.get("role", "")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            lines.append(f"用户：{content}")
        else:
            lines.append(f"助手：{content}")
    summary = (conv.get("summary") or "").strip()
    if summary:
        return f"【对话摘要】{summary}\n【最近对话】\n" + "\n".join(lines)
    return "【最近对话】\n" + "\n".join(lines)


def rewrite_question_if_needed(
    user_q: str,
    conv: Dict[str, Any],
    base_url: str,
    model: str,
    api_key: str,
) -> str:
    """
    If it's a follow-up question, rewrite it into a standalone query for retrieval.
    If rewrite fails, return original question.
    """
    if not is_followup(user_q):
        return user_q

    context = conv_context_text(conv, last_k_turns=4)
    system = (
        "你是一个问题改写器。给定对话上下文和用户追问，"
        "请把追问改写为一个独立、完整、适合用于教材检索的中文问题。"
        "不要回答问题，只输出改写后的问题本身。"
    )
    user = f"{context}\n\n用户追问：{user_q}\n\n改写后的独立问题："

    try:
        out = llm_chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout=35,
        )
        q2 = (out or "").strip().strip("“”\"")
        return q2 if q2 else user_q
    except Exception:
        return user_q


def retrieve_dual(
    index: Any,
    embedder: Any,
    q_raw: str,
    q_rewrite: str,
    topk: int = TOPK,
) -> List[Tuple[int, float]]:
    """
    Dual retrieval: search with raw and rewritten query, merge results by best score.
    Returns list of (chunk_id, score) sorted desc score.
    """
    def _search(q: str) -> List[Tuple[int, float]]:
        v = embed_query(embedder, q)
        D, I = index.search(v, topk)
        hits = []
        for idx, score in zip(I[0].tolist(), D[0].tolist()):
            if idx < 0:
                continue
            hits.append((int(idx), float(score)))
        return hits

    hits = _search(q_raw)
    if q_rewrite and q_rewrite != q_raw:
        hits += _search(q_rewrite)

    best: Dict[int, float] = {}
    for cid, sc in hits:
        if cid not in best or sc > best[cid]:
            best[cid] = sc

    merged = sorted(best.items(), key=lambda x: x[1], reverse=True)[:topk]
    return merged


def format_evidence_blocks(metas: List[ChunkMeta], hits: List[Tuple[int, float]]) -> Tuple[List[str], List[Dict[str, Any]]]:
    blocks = []
    refs = []
    for rank, (cid, score) in enumerate(hits, start=1):
        if cid < 0 or cid >= len(metas):
            continue
        m = metas[cid]
        excerpt = m.text
        # keep evidence block readable
        excerpt = excerpt[:900].strip()
        blocks.append(f"来源：{m.source_file} p{m.page_start}\n相似度：{score:.3f}\n内容：{excerpt}")
        refs.append(
            {
                "n": rank,
                "source_file": m.source_file,
                "page": f"{m.page_start}-{m.page_end}" if m.page_start != m.page_end else str(m.page_start),
                "score": round(score, 3),
            }
        )
    return blocks, refs


def answer_with_rag(
    user_q: str,
    conv: Dict[str, Any],
    index: Any,
    metas: List[ChunkMeta],
    embedder: Any,
    base_url: str,
    model: str,
    api_key: str,
) -> Tuple[str, Dict[str, Any]]:
    """
    Returns (answer_text, debug_info)
    """
    t0 = time.time()

    q_rewrite = rewrite_question_if_needed(user_q, conv, base_url, model, api_key)
    hits = retrieve_dual(index, embedder, user_q, q_rewrite, topk=TOPK)

    debug = {
        "q_raw": user_q,
        "q_rewrite": q_rewrite,
        "hits": hits,
    }

    if not hits or hits[0][1] < MIN_SCORE:
        # Evidence too weak. Still respond with helpful guidance and optionally general explanation.
        msg = (
            "【回答】\n\n"
            "资料不足：在当前已导入的 PDF 中未检索到与问题高度匹配的片段，无法基于教材给出可靠回答。\n\n"
            "建议：\n"
            "1）把问题说得更完整（例如把“它有什么作用”改成“指针有什么作用”）。\n"
            "2）确认教材是否包含该知识点，或导入更相关的讲义/教材。\n"
        )
        debug["elapsed_ms"] = int((time.time() - t0) * 1000)
        return msg, debug

    evidence_blocks, refs = format_evidence_blocks(metas, hits)
    context = conv_context_text(conv, last_k_turns=4)

    system = (
        "你是面向计算机专业课程的智能答疑助手。\n"
        "要求：\n"
        "1) 必须优先依据【证据】回答。\n"
        "2) 回答中涉及证据时，用 [1][2]... 标注引用编号（对应证据编号）。\n"
        "3) 若证据不足以覆盖问题的某部分，明确指出缺失点，并给出补充资料建议。\n"
        "4) 用中文回答，结构清晰。\n"
    )
    evidence_text = "\n\n".join([f"[{i+1}] {b}" for i, b in enumerate(evidence_blocks)])

    user = (
        f"{context}\n\n"
        f"【证据】\n{evidence_text}\n\n"
        f"【问题】{user_q}\n\n"
        f"请开始回答："
    )

    try:
        out = llm_chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout=45,
        )
        ans = out.strip()
    except Exception as e:
        # LLM down: fallback to evidence-only
        ans = (
            "【回答】\n\n"
            f"LLM 调用失败：{e}\n\n"
            "已返回检索到的资料摘要（未生成扩展解释）。\n\n"
        )
        for i, b in enumerate(evidence_blocks, start=1):
            ans += f"证据[{i}] 摘要：{b[:400]}...\n\n"

    # Append references and Top-K evidence
    ans += "\n\n【引用】\n"
    for r in refs:
        ans += f"[{r['n']}] {r['source_file']} p{r['page']}（相似度 {r['score']}）\n"

    ans += "\n【本次检索到的证据（Top-K）】\n"
    for i, (cid, score) in enumerate(hits, start=1):
        m = metas[cid]
        ans += f"\n[{i}] 相似度：{score:.3f}｜{m.source_file} p{m.page_start}\n{m.text[:600].strip()}\n"

    debug["elapsed_ms"] = int((time.time() - t0) * 1000)
    return ans, debug


# ----------------------------
# Conversation Store
# ----------------------------
def init_state() -> None:
    if "conv_store" not in st.session_state:
        st.session_state.conv_store = {}  # conv_id -> {title, messages, summary, created_ms}
    if "active_conv_id" not in st.session_state:
        st.session_state.active_conv_id = ""
    if "debug_mode" not in st.session_state:
        st.session_state.debug_mode = False


def new_conversation() -> str:
    cid = sha1_text(str(now_ms()) + str(time.time()))
    st.session_state.conv_store[cid] = {
        "title": "未命名对话",
        "messages": [],
        "summary": "",
        "created_ms": now_ms(),
    }
    st.session_state.active_conv_id = cid
    return cid


def get_active_conv() -> Dict[str, Any]:
    cid = st.session_state.active_conv_id
    if not cid or cid not in st.session_state.conv_store:
        cid = new_conversation()
    return st.session_state.conv_store[cid]


def set_conv_title_if_needed(conv: Dict[str, Any], first_user_text: str) -> None:
    if conv.get("title") == "未命名对话":
        conv["title"] = short_title_from_text(first_user_text, max_len=18)


def summarize_old_messages(
    old_msgs: List[Dict[str, str]],
    base_url: str,
    model: str,
    api_key: str,
) -> str:
    """
    Summarize old messages into a short Chinese summary.
    If LLM fails, return empty string.
    """
    if not old_msgs:
        return ""

    transcript = []
    for m in old_msgs:
        role = m.get("role", "")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        prefix = "用户" if role == "user" else "助手"
        transcript.append(f"{prefix}：{content}")
    text = "\n".join(transcript)
    if not text:
        return ""

    system = "你是对话摘要器。请把给定的对话内容压缩成 5-8 条要点，用中文输出。"
    user = f"对话内容：\n{text}\n\n请输出摘要要点："

    try:
        out = llm_chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout=35,
        )
        return out.strip()
    except Exception:
        return ""


def enforce_turn_limit(
    conv: Dict[str, Any],
    max_turns: int,
    base_url: str,
    model: str,
    api_key: str,
) -> None:
    """
    Keep last max_turns user questions and assistant answers (approx 2*max_turns messages).
    Optionally summarize trimmed part into conv["summary"].
    """
    msgs = conv.get("messages", [])
    if not msgs:
        return

    keep_n = max_turns * 2
    if len(msgs) <= keep_n:
        return

    old = msgs[:-keep_n]
    new = msgs[-keep_n:]

    if AUTO_SUMMARIZE and base_url and model and api_key:
        add_sum = summarize_old_messages(old, base_url, model, api_key)
        if add_sum:
            prev = (conv.get("summary") or "").strip()
            conv["summary"] = (prev + "\n" + add_sum).strip() if prev else add_sum

    conv["messages"] = new


# ----------------------------
# UI
# ----------------------------
def sidebar_ui(base_url: str, model: str, api_key: str) -> Dict[str, Any]:
    st.sidebar.title("对话")
    if st.sidebar.button("➕ 新聊天", use_container_width=True):
        new_conversation()
        st.rerun()

    # Conversation list (expanded)
    conv_store: Dict[str, Any] = st.session_state.conv_store
    # sort by created time desc
    items = sorted(conv_store.items(), key=lambda kv: kv[1].get("created_ms", 0), reverse=True)

    if not items:
        new_conversation()
        items = [(st.session_state.active_conv_id, st.session_state.conv_store[st.session_state.active_conv_id])]

    for cid, conv in items:
        title = conv.get("title", "未命名对话")
        is_active = (cid == st.session_state.active_conv_id)
        label = f"• {title}" if is_active else title
        if st.sidebar.button(label, key=f"convbtn_{cid}", use_container_width=True):
            st.session_state.active_conv_id = cid
            st.rerun()

    st.sidebar.divider()

    # Minimal status
    with st.sidebar.expander("状态", expanded=False):
        st.write(f"Embedding：`{EMBED_MODEL_NAME}`")
        st.write(f"对话轮数：默认 `{MAX_TURNS_DEFAULT}`")
        if base_url and model and api_key:
            st.success("LLM 配置：已检测到")
            st.code(f"BASE={base_url}\nMODEL={model}", language="text")
        else:
            st.warning("LLM 配置：未完整（部署时建议用 Streamlit secrets 设置）")

    # Debug toggle
    st.session_state.debug_mode = st.sidebar.checkbox("Debug", value=st.session_state.debug_mode)

    # Admin (optional) - hidden unless server has ADMIN_TOKEN
    admin_token_server = get_env_or_secret(ENV_ADMIN, "")
    if admin_token_server:
        with st.sidebar.expander("管理员", expanded=False):
            token_input = st.text_input("ADMIN_TOKEN", type="password", key="admin_token_input")
            ok = token_input and token_input.strip() == admin_token_server.strip()
            if ok:
                st.success("管理员验证通过")
                if st.button("重建索引（扫描 data/pdf）", use_container_width=True):
                    st.session_state._force_rebuild = True
                    st.rerun()
            else:
                st.info("输入正确的 ADMIN_TOKEN 才会显示管理操作")

    return {}


def render_chat(conv: Dict[str, Any]) -> None:
    for m in conv.get("messages", []):
        role = m.get("role", "")
        content = m.get("content", "")
        if role not in ("user", "assistant"):
            continue
        with st.chat_message("user" if role == "user" else "assistant"):
            st.markdown(content)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    init_state()

    # LLM config (secrets/env only; no UI inputs to avoid leaking key on public app)
    base_url = get_env_or_secret(ENV_BASE, "").strip()
    model = get_env_or_secret(ENV_MODEL, "").strip()
    api_key = get_env_or_secret(ENV_KEY, "").strip()

    sidebar_ui(base_url, model, api_key)

    st.title(APP_TITLE)
    st.caption("方案 A：将教材 PDF 放入仓库 data/pdf/，部署后所有用户可直接提问。")

    conv = get_active_conv()

    # Load embedder + index
    embedder = load_embedder(EMBED_MODEL_NAME)

    force = bool(st.session_state.get("_force_rebuild", False))
    st.session_state["_force_rebuild"] = False

    with st.spinner("检查/加载索引..."):
        try:
            idx, metas, status = ensure_index_ready(embedder, force_rebuild=force)
        except Exception as e:
            idx, metas, status = None, [], f"索引初始化失败：{type(e).__name__}: {e}"

    st.info(status)

    # Chat history
    render_chat(conv)

    # Input
    user_q = st.chat_input("请输入你的问题（支持多轮追问）")
    if user_q is None:
        return

    user_q = user_q.strip()
    if not user_q:
        return

    # Append user message
    conv["messages"].append({"role": "user", "content": user_q})
    set_conv_title_if_needed(conv, user_q)

    with st.chat_message("user"):
        st.markdown(user_q)

    # Answer
    with st.chat_message("assistant"):
        if idx is None or not metas:
            ans = "【回答】\n\n索引未就绪：请确认 data/pdf/ 下存在可提取文本的 PDF。"
            st.markdown(ans)
            conv["messages"].append({"role": "assistant", "content": ans})
        else:
            if not (base_url and model and api_key):
                ans = (
                    "【回答】\n\n"
                    "LLM 配置不完整：部署时请在 Streamlit secrets 中设置：\n"
                    "- LLM_BASE_URL\n- LLM_MODEL\n- LLM_API_KEY\n"
                    "当前仅能完成检索与证据展示，无法生成扩展解释。"
                )
                # still show evidence
                q_rewrite = user_q
                hits = retrieve_dual(idx, embedder, user_q, q_rewrite, topk=TOPK)
                if hits and hits[0][1] >= MIN_SCORE:
                    evidence_blocks, refs = format_evidence_blocks(metas, hits)
                    ans += "\n\n【本次检索到的证据（Top-K）】\n"
                    for i, (cid, score) in enumerate(hits, start=1):
                        m = metas[cid]
                        ans += f"\n[{i}] 相似度：{score:.3f}｜{m.source_file} p{m.page_start}\n{m.text[:600].strip()}\n"
                st.markdown(ans)
                conv["messages"].append({"role": "assistant", "content": ans})
            else:
                with st.spinner("检索 + 生成中..."):
                    ans, debug = answer_with_rag(
                        user_q=user_q,
                        conv=conv,
                        index=idx,
                        metas=metas,
                        embedder=embedder,
                        base_url=base_url,
                        model=model,
                        api_key=api_key,
                    )
                st.markdown(ans)
                conv["messages"].append({"role": "assistant", "content": ans})

                # Enforce default 12 turns
                enforce_turn_limit(conv, MAX_TURNS_DEFAULT, base_url, model, api_key)

                # Debug panel
                if st.session_state.debug_mode:
                    with st.expander("Debug 信息", expanded=True):
                        st.json(debug, expanded=True)

    # persist in session_state
    st.session_state.conv_store[st.session_state.active_conv_id] = conv


if __name__ == "__main__":
    main()
