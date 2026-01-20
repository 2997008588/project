import os
import re
import json
import time
import pickle
from dataclasses import dataclass
from typing import List, Dict, Tuple
from collections import Counter

import numpy as np
import fitz  # pymupdf
import faiss
import requests
import streamlit as st
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ---------------------------
# Config
# ---------------------------
DATA_DIR = "data"
PDF_DIR = os.path.join(DATA_DIR, "pdf")
INDEX_DIR = os.path.join(DATA_DIR, "index")
FAISS_PATH = os.path.join(INDEX_DIR, "index.faiss")
META_PATH = os.path.join(INDEX_DIR, "meta.pkl")

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "BAAI/bge-small-zh-v1.5")
TOP_K = 6
CHUNK_SIZE = 700      # approx chars for Chinese text
CHUNK_OVERLAP = 120   # overlap chars
MIN_SCORE = 0.18      # similarity threshold; adjust if needed

LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip()
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").strip().rstrip("/")
LLM_MODEL = os.getenv("LLM_MODEL", "").strip()

SYSTEM_PROMPT = """你是面向计算机专业课程的智能答疑助手。
你只能基于【证据】回答，不得编造证据中不存在的内容。
如果证据不足以支持回答，请明确说“资料不足”，并建议用户补充资料或换一种问法。

输出格式必须为：
1) 【回答】用要点分条陈述
2) 【引用】列出你使用到的证据编号与来源（文件名、页码），例如：[1] 文件.pdf p12

引用要求（重要）：
- 【引用】中的文件名与页码，必须来自【证据】里每条的“来源：...”字段，不要写成“文件.pdf”这类泛化名称。
"""

# ---------------------------
# Topic hint (lightweight)
# ---------------------------
DOMAIN_KEYWORDS = {
    "C/C++ 编程": [
        "c语言", "c 程序", "c程序", "指针", "数组", "函数", "结构体", "宏", "预处理", "编译",
        "printf", "scanf", "malloc", "free", "stdio", "stdlib", "pointer", "array",
        "数组指针", "指针数组", "二维数组", "多维数组"
    ],
    "计算机网络": [
        "tcp", "udp", "ip", "路由", "arp", "mac", "dns", "拥塞", "滑动窗口", "csma", "以太网",
        "三次握手", "四次挥手"
    ],
    "操作系统": [
        "进程", "线程", "调度", "死锁", "内存", "分页", "段页", "虚拟存储", "文件系统",
        "linux", "shell", "系统调用"
    ],
    "软件工程/测试": [
        "需求", "用例", "测试", "junit", "集成测试", "验收测试", "螺旋模型", "v模型", "sdlc"
    ],
    "人工智能/机器学习": [
        "机器学习", "深度学习", "神经网络", "回归", "分类", "聚类", "损失", "梯度", "lstm", "cnn"
    ],
    "音乐": [
        "古典音乐", "交响乐", "奏鸣曲", "协奏曲", "巴洛克", "浪漫主义", "古典主义",
        "莫扎特", "贝多芬", "巴赫", "柴可夫斯基", "作曲家", "钢琴", "小提琴"
    ],
}

def infer_topic(text: str) -> Tuple[str, int]:
    """Return (topic_name, score). Score is keyword hit count."""
    t = (text or "").lower()
    best_topic = "未知"
    best_score = 0
    for topic, kws in DOMAIN_KEYWORDS.items():
        score = 0
        for k in kws:
            if not k:
                continue
            if k.lower() in t:
                score += 1
        if score > best_score:
            best_score = score
            best_topic = topic
    return best_topic, best_score

def infer_query_topic(query: str) -> str:
    topic, score = infer_topic(query)
    return topic if score > 0 else "未知"

def infer_corpus_topic(chunks: List["Chunk"], sample_n: int = 80) -> str:
    if not chunks:
        return "未知"
    n = min(sample_n, len(chunks))
    step = max(1, len(chunks) // n)
    sample_text = "\n".join(chunks[i].text for i in range(0, len(chunks), step))[:200000]
    topic, score = infer_topic(sample_text)
    return topic if score > 0 else "未知"

# ---------------------------
# Data structures
# ---------------------------
@dataclass
class Chunk:
    chunk_id: str
    file_name: str
    page_start: int
    page_end: int
    text: str

# ---------------------------
# Utility
# ---------------------------
def ensure_dirs():
    os.makedirs(PDF_DIR, exist_ok=True)
    os.makedirs(INDEX_DIR, exist_ok=True)

def clean_text(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = s.replace("-\n", "")
    return s.strip()

def extract_pdf_pages(pdf_path: str) -> List[Tuple[int, str]]:
    doc = fitz.open(pdf_path)
    pages = []
    for i in range(len(doc)):
        text = doc[i].get_text("text") or ""
        text = clean_text(text)
        if text:
            pages.append((i + 1, text))
    doc.close()
    return pages

def chunk_pages(file_name: str, pages: List[Tuple[int, str]],
                chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Chunk]:
    chunks: List[Chunk] = []
    buf = ""
    buf_page_start = None
    buf_page_end = None
    chunk_idx = 0

    def flush_buffer(final: bool = False):
        nonlocal buf, buf_page_start, buf_page_end, chunk_idx
        txt = clean_text(buf)
        if txt:
            chunk_id = f"{file_name}::c{chunk_idx:06d}"
            chunks.append(Chunk(
                chunk_id=chunk_id,
                file_name=file_name,
                page_start=buf_page_start or 1,
                page_end=buf_page_end or (buf_page_start or 1),
                text=txt
            ))
            chunk_idx += 1
        if final:
            buf = ""
            buf_page_start = None
            buf_page_end = None
        else:
            buf = txt[-overlap:] if len(txt) > overlap else txt
            buf_page_start = buf_page_end

    for page_no, page_text in pages:
        if buf_page_start is None:
            buf_page_start = page_no
        buf_page_end = page_no

        paras = [p.strip() for p in page_text.split("\n") if p.strip()]
        for p in paras:
            if not buf:
                buf_page_start = page_no
            buf_page_end = page_no

            candidate = (buf + "\n" + p).strip() if buf else p
            if len(candidate) <= chunk_size:
                buf = candidate
            else:
                flush_buffer(final=False)
                buf = p
                buf_page_start = page_no
                buf_page_end = page_no

                while len(buf) > chunk_size:
                    part = buf[:chunk_size]
                    buf = buf[chunk_size:]
                    chunk_id = f"{file_name}::c{chunk_idx:06d}"
                    chunks.append(Chunk(
                        chunk_id=chunk_id,
                        file_name=file_name,
                        page_start=page_no,
                        page_end=page_no,
                        text=clean_text(part)
                    ))
                    chunk_idx += 1
                    buf = (part[-overlap:] + buf) if overlap > 0 else buf

    if buf.strip():
        flush_buffer(final=True)

    return chunks

def load_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL_NAME)

def embed_texts(embedder: SentenceTransformer, texts: List[str], batch_size: int = 32) -> np.ndarray:
    vecs = embedder.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    return np.asarray(vecs, dtype="float32")

def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index

def save_index(index: faiss.Index, chunks: List[Chunk]):
    faiss.write_index(index, FAISS_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(chunks, f)

def load_index() -> Tuple[faiss.Index, List[Chunk]]:
    index = faiss.read_index(FAISS_PATH)
    with open(META_PATH, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

# ---------------------------
# LLM
# ---------------------------
def llm_chat(messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
    if not (LLM_API_KEY and LLM_BASE_URL and LLM_MODEL):
        raise RuntimeError("LLM_API_KEY / LLM_BASE_URL / LLM_MODEL 未配置完整。")

    if LLM_BASE_URL.endswith("/v1"):
        url = f"{LLM_BASE_URL}/chat/completions"
    else:
        url = f"{LLM_BASE_URL}/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=90)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]

# ---------------------------
# Multi-turn helpers
# ---------------------------
FOLLOWUP_MARKERS = [
    "它", "他", "她", "这", "那", "这个", "那个", "上面", "刚才", "继续", "再说",
    "为什么", "怎么", "那然后", "然后呢", "还有呢", "如何", "能不能"
]

def is_followup_question(q: str) -> bool:
    q = (q or "").strip()
    if len(q) <= 10:
        return True
    return any(m in q for m in FOLLOWUP_MARKERS)

def build_retrieval_query(current_q: str, user_history: List[str], max_turns: int = 4) -> str:
    current_q = (current_q or "").strip()
    if not user_history:
        return current_q
    if not is_followup_question(current_q):
        return current_q
    tail = [h.strip() for h in user_history[-max_turns:] if h.strip()]
    return "；".join(tail + [current_q])

def format_dialog_context(history: List[Dict[str, str]], max_pairs: int = 3, max_chars_each: int = 300) -> str:
    if not history:
        return ""
    tail = history[-max_pairs*2:]
    lines = []
    for m in tail:
        role = m.get("role", "")
        content = (m.get("content", "") or "").strip()
        if not content:
            continue
        if len(content) > max_chars_each:
            content = content[:max_chars_each] + "…"
        if role == "user":
            lines.append(f"用户：{content}")
        elif role == "assistant":
            lines.append(f"助手：{content}")
    return "\n".join(lines)

# ---------------------------
# Retrieval + Guards
# ---------------------------
def retrieve(index: faiss.Index, chunks: List[Chunk], embedder: SentenceTransformer,
             query: str, top_k: int = TOP_K) -> List[Tuple[float, Chunk]]:
    qv = embed_texts(embedder, [query], batch_size=1)
    scores, ids = index.search(qv, top_k)
    results = []
    for s, i in zip(scores[0], ids[0]):
        if i < 0:
            continue
        results.append((float(s), chunks[int(i)]))
    return results

def format_evidence(results: List[Tuple[float, Chunk]]) -> str:
    lines = []
    for idx, (score, c) in enumerate(results, start=1):
        src = f"{c.file_name} p{c.page_start}" if c.page_start == c.page_end else f"{c.file_name} p{c.page_start}-{c.page_end}"
        snippet = c.text[:450].replace("\n", " ")
        lines.append(f"[{idx}] 来源：{src}\n片段：{snippet}")
    return "\n\n".join(lines)

STOPWORDS = set([
    "它", "他", "她", "这", "那", "这个", "那个", "上面", "刚才", "继续", "再说",
    "为什么", "怎么", "如何", "能不能", "可以吗", "吗", "呢", "呀", "啊",
    "关系", "联系", "区别", "对比", "比较", "相关", "之间", "以及", "还有", "然后",
    "什么", "是啥", "是什么", "是什么呢"
])

def extract_keywords(q: str) -> List[str]:
    """
    改进版：
    1) 先从领域关键词里提取（更稳，避免抽成“数组有什么关系”这种长串）
    2) 再做通用分词抽取兜底
    """
    q_raw = (q or "").strip()
    q_low = q_raw.lower()

    kws: List[str] = []

    # 1) 领域关键词优先抽取（长度>=2，避免过碎）
    for topic_kws in DOMAIN_KEYWORDS.values():
        for k in topic_kws:
            kk = (k or "").strip().lower()
            if len(kk) >= 2 and kk in q_low and kk not in kws:
                kws.append(kk)

    # 2) 通用抽取（去掉常见虚词）
    q2 = q_low
    for pat in [
        "是什么呢", "是什么", "是啥", "什么", "怎么", "如何", "为什么",
        "能不能", "可以吗", "吗", "呢", "呀", "啊", "的", "了"
    ]:
        q2 = q2.replace(pat, "")

    words = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z]{3,}", q2)
    for w in words:
        w = w.strip().lower()
        if not w or w in STOPWORDS:
            continue
        # 进一步剔除纯关系类词
        if w in ["有什么关系", "有何关系", "有什么联系", "有何联系"]:
            continue
        if w not in kws:
            kws.append(w)

    return kws

def evidence_has_keywords(results: List[Tuple[float, Chunk]], query: str, min_hits: int = 1) -> bool:
    kws = extract_keywords(query)
    if not kws:
        return True
    evidence_text = "\n".join(c.text.lower() for _, c in results)
    hits = sum(1 for k in kws if k in evidence_text)
    return hits >= min_hits

def make_hint_block(query: str, corpus_topic: str) -> str:
    q_topic = infer_query_topic(query)
    if q_topic != "未知" and corpus_topic != "未知" and q_topic != corpus_topic:
        return (
            "\n\n【提示】\n"
            f"- 检测到你的问题更像是「{q_topic}」相关，但当前已导入资料主要是「{corpus_topic}」。\n"
            f"- 建议导入与「{q_topic}」相关的教材/讲义/课件 PDF，或换成更贴合当前资料的问法。"
        )
    return ""

def fallback_answer(query: str, results: List[Tuple[float, Chunk]], err: Exception) -> str:
    lines = [
        "【回答】",
        f"- LLM 调用失败：{type(err).__name__}({err})",
        "- 已返回检索到的资料摘要（未生成扩展解释）。",
        "",
        "【引用】"
    ]
    if not results:
        lines.append("- 无")
        return "\n".join(lines)

    for i, (score, c) in enumerate(results, start=1):
        src = f"{c.file_name} p{c.page_start}" if c.page_start == c.page_end else f"{c.file_name} p{c.page_start}-{c.page_end}"
        lines.append(f"- [{i}] {src}")

    lines.append("")
    for i, (score, c) in enumerate(results, start=1):
        snippet = c.text.strip().replace("\n", " ")
        snippet = snippet[:260] + ("…" if len(snippet) > 260 else "")
        lines.append(f"证据[{i}] 摘要：{snippet}")
    return "\n".join(lines)

def answer_with_rag(
    query: str,
    results: List[Tuple[float, Chunk]],
    corpus_topic: str,
    dialog_context: str = "",
    guard_query: str = ""
) -> str:
    # 1) 相似度阈值过滤
    if (not results) or (max(s for s, _ in results) < MIN_SCORE):
        base = (
            "【回答】\n"
            "- 资料不足：在当前已导入的PDF资料中未检索到足以支撑该问题的明确依据。\n\n"
            "【引用】\n"
            "- 无\n\n"
            "建议：换一种更具体的问法，或导入包含该知识点的讲义/教材章节。"
        )
        return base + make_hint_block(query, corpus_topic)

    # 2) 关键词命中检查：这里改为用 guard_query（多轮时=检索用 query），避免追问句误杀
    guard_q = guard_query.strip() if guard_query else query
    if not evidence_has_keywords(results, guard_q, min_hits=1):
        base = (
            "【回答】\n"
            "- 资料不足：检索到的片段与问题关键词不匹配，无法基于现有PDF给出可靠回答。\n\n"
            "【引用】\n"
            "- 无\n\n"
            "建议：导入包含该知识点的教材/讲义，或换更贴合本资料的问法。"
        )
        return base + make_hint_block(query, corpus_topic)

    evidence = format_evidence(results)

    context_block = ""
    if dialog_context.strip():
        context_block = f"""【对话上下文（仅供理解指代，不作为证据）】
{dialog_context}

"""

    user_prompt = f"""{context_block}【问题】
{query}

【证据】
{evidence}
"""

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.append({"role": "user", "content": user_prompt})
    return llm_chat(messages)

# ---------------------------
# Streamlit App
# ---------------------------
def main():
    ensure_dirs()
    st.set_page_config(page_title="PDF+FAISS 课程答疑（RAG）", layout="wide")
    st.title("课程智能答疑（PDF + FAISS + API LLM）")

    with st.sidebar:
        st.header("索引管理")
        st.write("将PDF放入：`data/pdf/`")
        rebuild = st.button("重建索引（扫描所有PDF）")
        st.divider()
        st.header("LLM 配置检查")
        st.write(f"BASE_URL: {'已设置' if LLM_BASE_URL else '未设置'}")
        st.write(f"MODEL: {'已设置' if LLM_MODEL else '未设置'}")
        st.write(f"API_KEY: {'已设置' if LLM_API_KEY else '未设置'}")
        st.caption("如未设置，请先配置环境变量：LLM_API_KEY / LLM_BASE_URL / LLM_MODEL")

        st.divider()
        st.header("多轮对话设置")
        enable_multiturn = st.checkbox("启用多轮对话（上下文跟随）", value=True)
        memory_turns = st.slider("记忆的用户提问轮数", min_value=1, max_value=12, value=4, step=1)
        show_retrieval_query = st.checkbox("显示本轮实际用于检索的问题", value=True)
        if st.button("清空对话历史"):
            st.session_state.history = []
            st.rerun()

    if "embedder" not in st.session_state:
        with st.spinner("加载 embedding 模型..."):
            st.session_state.embedder = load_embedder()

    index_exists = os.path.exists(FAISS_PATH) and os.path.exists(META_PATH)

    def do_build():
        pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
        if not pdf_files:
            st.error("未找到PDF。请将PDF放入 data/pdf/ 后重试。")
            return

        all_chunks: List[Chunk] = []
        for pf in tqdm(pdf_files, desc="PDF解析"):
            p = os.path.join(PDF_DIR, pf)
            pages = extract_pdf_pages(p)
            chunks = chunk_pages(pf, pages)
            all_chunks.extend(chunks)

        if not all_chunks:
            st.error("PDF未抽取到有效文本（可能是扫描版）。请更换文本型PDF或后续加OCR。")
            return

        texts = [c.text for c in all_chunks]
        with st.spinner("向量化中..."):
            vecs = embed_texts(st.session_state.embedder, texts, batch_size=32)

        with st.spinner("建立FAISS索引..."):
            index = build_faiss_index(vecs)
            save_index(index, all_chunks)

        st.success(f"索引构建完成：chunks={len(all_chunks)}，PDF={len(pdf_files)}")

    if rebuild or (not index_exists):
        if not index_exists:
            st.info("未检测到索引，将自动构建一次。")
        do_build()

    if not (os.path.exists(FAISS_PATH) and os.path.exists(META_PATH)):
        st.stop()

    if "index" not in st.session_state or "chunks" not in st.session_state:
        with st.spinner("加载索引..."):
            st.session_state.index, st.session_state.chunks = load_index()

    if "corpus_topic" not in st.session_state:
        st.session_state.corpus_topic = infer_corpus_topic(st.session_state.chunks)

    with st.sidebar:
        st.divider()
        st.header("资料主题推断")
        st.write("当前资料更像：", st.session_state.corpus_topic)

    if "history" not in st.session_state:
        st.session_state.history = []

    # 渲染历史
    for m in st.session_state.history:
        role = m.get("role", "")
        content = m.get("content", "")
        if role in ("user", "assistant") and content:
            st.chat_message(role).write(content)

    query = st.chat_input("请输入你的问题（基于已导入PDF资料回答）")
    if query:
        st.chat_message("user").write(query)

        # 构造检索 query（多轮关键）
        user_questions = [m["content"] for m in st.session_state.history if m.get("role") == "user"]
        retrieval_query = query
        if enable_multiturn:
            retrieval_query = build_retrieval_query(query, user_questions, max_turns=memory_turns)

        if show_retrieval_query and retrieval_query != query:
            st.info(f"本轮实际用于检索的问题：{retrieval_query}")

        t0 = time.time()
        results = retrieve(st.session_state.index, st.session_state.chunks, st.session_state.embedder, retrieval_query, TOP_K)
        t_retr = time.time()

        dialog_context = format_dialog_context(st.session_state.history, max_pairs=3) if enable_multiturn else ""

        try:
            # 关键改动：guard_query 用 retrieval_query，避免“追问句关键词不匹配”误杀
            ans = answer_with_rag(
                query=query,
                results=results,
                corpus_topic=st.session_state.corpus_topic,
                dialog_context=dialog_context,
                guard_query=retrieval_query
            )
        except Exception as e:
            ans = fallback_answer(query, results, e)

        t_llm = time.time()
        st.chat_message("assistant").write(ans)

        st.session_state.history.append({"role": "user", "content": query})
        st.session_state.history.append({"role": "assistant", "content": ans})
        st.session_state.history = st.session_state.history[-24:]

        with st.expander("本次检索到的证据（Top-K）", expanded=True):
            cnt = Counter(c.file_name for _, c in results)
            if cnt:
                dist = "；".join([f"{k}×{v}" for k, v in cnt.items()])
                st.write("命中文件分布：", dist)

            for i, (score, c) in enumerate(results, start=1):
                src = f"{c.file_name} p{c.page_start}" if c.page_start == c.page_end else f"{c.file_name} p{c.page_start}-{c.page_end}"
                st.markdown(f"**[{i}] 相似度：{score:.3f}｜{src}｜{c.chunk_id}**")
                st.write(c.text)

        st.caption(f"检索耗时：{(t_retr - t0):.2f}s｜生成耗时：{(t_llm - t_retr):.2f}s")

if __name__ == "__main__":
    main()
