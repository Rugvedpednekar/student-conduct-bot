# app/main.py
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from .rag import retrieve, format_context, build_prompt, answer_with_llm, build_index

app = FastAPI(title="UHart Conduct Q&A (Gemini)", version="0.1.0")

# --- CORS (allow the browser JS to call the API) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten later if you want
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Serve /web/* from your repo's web/ folder ---
ROOT = Path(__file__).resolve().parents[1]       # project root
WEB_DIR = ROOT / "web"                           # D:\student-conduct-bot\web
app.mount("/web", StaticFiles(directory=str(WEB_DIR), html=True), name="web")

# Optional: redirect "/" to the UI
@app.get("/")
def root():
    return RedirectResponse(url="/web/index.html")

# --- Simple endpoints already in your project ---
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/emergency")
def emergency():
    return {
        "emergency": True,
        "public_safety": "860-768-7777",
        "note": "If this is an emergency, call Public Safety or 911."
    }

from pydantic import BaseModel
class Query(BaseModel):
    question: str

@app.post("/ask")
def ask(q: Query):
    # retrieve -> format context -> prompt -> answer
    pairs = retrieve(q.question, k=6)
    ctx = format_context(pairs)
    prompt = build_prompt(q.question, ctx)
    answer = answer_with_llm(prompt)

    # send back light-weight citation objects
    citations = [{"page": m["page"], "source": m["source"], "url": m["url"]} for _, m in pairs]
    return {"answer": answer, "citations": citations}

# Build index on import (or call build_index() from ingest)
try:
    build_index()
except Exception as e:
    # don’t crash API if index build fails; you can still hit /health
    print(f"[warn] build_index() skipped: {e}")
