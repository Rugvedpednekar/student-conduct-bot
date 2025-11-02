# app/rag.py
import io, re, os, json, requests
from pathlib import Path
from typing import List, Tuple
import numpy as np
from pypdf import PdfReader

from .config import HANDBOOK_URL, USE_GEMINI_EMBEDS, USE_GEMINI_LLM
from .guardrails import SYSTEM_INSTRUCTIONS, REFUSAL




# -------------------------------
# Simple on-disk index (no Chroma)
# -------------------------------
INDEX_DIR = Path("./.simple_index")
EMB_PATH  = INDEX_DIR / "embeddings.npy"
DOCS_PATH = INDEX_DIR / "documents.json"
METAS_PATH= INDEX_DIR / "metadatas.json"

# -------------------------------
# Embeddings (Gemini)
# -------------------------------
if USE_GEMINI_EMBEDS:
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    def embed(texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        B = 8
        total = (len(texts) + B - 1) // B
        for i in range(0, len(texts), B):
            batch = texts[i:i+B]
            print(f"[embed] batch {i//B + 1}/{total} (size={len(batch)})")
            try:
                resp = genai.embed_content(model="models/text-embedding-004", content=batch)
                vecs = resp["embedding"]
            except Exception as e:
                print(f"[ERROR in Gemini Embedding]: {e}")
                # Fail soft with zero-vectors so app still returns a message
                vecs = [[0.0]*768 for _ in batch]
            out.extend(vecs)
        return out

else:
    raise RuntimeError("Set USE_GEMINI_EMBEDS=True in config.py")

# -------------------------------
# PDF → pages → chunks
# -------------------------------
def _pdf_pages(url: str):
    print(f"[ingest] fetching PDF: {url}")
    data = requests.get(url, timeout=60).content
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for i, p in enumerate(reader.pages, start=1):
        text = p.extract_text() or ""
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            pages.append((i, text))
    print(f"[ingest] loaded {len(pages)} PDF pages with text")
    return pages

def _chunk(text: str, size: int = 1200, overlap: int = 150):
    """
    Safe chunker:
    - Always advances by a positive step
    - Stops at end of text
    - Adjusts if overlap >= size
    """
    if not text:
        return []

    size = max(1, int(size))
    overlap = max(0, int(overlap))
    if overlap >= size:
        overlap = size // 3  # ensure positive step

    chunks = []
    start = 0
    L = len(text)
    step = size - overlap  # >= 1

    while start < L:
        end = min(L, start + size)
        chunks.append(text[start:end])
        if end >= L:
            break
        new_start = start + step
        if new_start <= start:
            new_start = start + 1
        start = new_start
    return chunks

# -------------------------------
# Build / Load index
# -------------------------------
_mem = {"E": None, "docs": None, "metas": None}

def build_index():
    """
    Build a simple cosine-similarity index:
    - documents.json : list[str] chunks (UTF-8)
    - metadatas.json : list[dict] with page/source/url (UTF-8)
    - embeddings.npy : float32 array (L2-normalized)
    """
    INDEX_DIR.mkdir(exist_ok=True)
    if EMB_PATH.exists() and DOCS_PATH.exists() and METAS_PATH.exists():
        print("[index] already exists; skipping rebuild")
        return

    pages = _pdf_pages(HANDBOOK_URL)
    docs, metas = [], []
    total_chunks = 0

    for page_num, text in pages:
        page_chunks = _chunk(text)
        if not page_chunks:
            continue
        docs.extend(page_chunks)
        metas.extend(
            {"page": page_num, "source": "Student Handbook", "url": HANDBOOK_URL}
            for _ in page_chunks
        )
        total_chunks += len(page_chunks)

    if not docs:
        raise RuntimeError("No text extracted from handbook. Check HANDBOOK_URL and PDF content.")

    print(f"[index] total chunks: {total_chunks}")
    vecs = np.array(embed(docs), dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    vecs = vecs / norms

    np.save(EMB_PATH, vecs)
    DOCS_PATH.write_text(json.dumps(docs, ensure_ascii=False), encoding="utf-8")
    METAS_PATH.write_text(json.dumps(metas, ensure_ascii=False), encoding="utf-8")
    print("[index] saved embeddings and metadata to .simple_index/")

def _load_index():
    if _mem["E"] is None:
        print("[index] loading vectors into memory…")
        if not (EMB_PATH.exists() and DOCS_PATH.exists() and METAS_PATH.exists()):
            raise RuntimeError(
                "Simple index incomplete. Re-run: py -3.13 -m app.ingest "
                "(need embeddings.npy, documents.json, metadatas.json)"
            )
        _mem["E"] = np.load(EMB_PATH)
        _mem["docs"] = json.loads(DOCS_PATH.read_text(encoding="utf-8"))
        _mem["metas"] = json.loads(METAS_PATH.read_text(encoding="utf-8"))
        print(f"[index] loaded {_mem['E'].shape[0]} vectors")


# -------------------------------
# Retrieval
# -------------------------------
def retrieve(query: str, k=6) -> List[Tuple[str, dict]]:
    _load_index()
    qv = np.array(embed([query])[0], dtype=np.float32)
    qv = qv / (np.linalg.norm(qv) + 1e-12)
    sims = _mem["E"] @ qv  # cosine similarity
    topk_idx = sims.argsort()[-k:][::-1]
    return [(_mem["docs"][i], _mem["metas"][i]) for i in topk_idx]

def format_context(pairs):
    parts = []
    for d, m in pairs:
        parts.append(f"{d}\n(Source: {m['source']}, p.{m['page']}) <{m['url']}>")
    return "\n\n---\n\n".join(parts)

def build_prompt(user_q: str, context: str):
    return f"""{SYSTEM_INSTRUCTIONS}

QUESTION:
{user_q}

CONTEXT:
{context}

INSTRUCTIONS:
- Answer ONLY from CONTEXT.
- Include page citations like (Student Handbook, p.X) and the link already provided.
- If missing, respond with this exact refusal block:

{REFUSAL}

RESPONSE:
"""

# -------------------------------
# LLM (Gemini 1.5)
# -------------------------------
def answer_with_llm(prompt: str) -> str:
    if not USE_GEMINI_LLM:
        return "LLM not configured; enable USE_GEMINI_LLM."
    import google.generativeai as genai

    # Change the model name to a current, stable alias
    # "gemini-2.5-flash" is the current recommended alias
    model_name = "gemini-2.5-flash"

    try:
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=SYSTEM_INSTRUCTIONS,
        )
        resp = model.generate_content(
            [{"role": "user", "parts": [prompt]}],
            generation_config={"temperature": 0.1}
        )
        text = getattr(resp, "text", "") or ""
        if not text.strip():
            print(f"[WARN] Empty LLM text for model {model_name}: {resp}")
            return "I couldn’t generate a response from the context provided."
        return text.strip()
    except Exception as e:
        # The error message in the console will now reflect the new model name
        print(f"[ERROR in Gemini LLM {model_name}]: {e}")
        return "Sorry, I had trouble generating a response. Please try again."
