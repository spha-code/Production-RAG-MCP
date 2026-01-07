# backend/local_llm.py
import os
import time
import requests
from functools import lru_cache
from dotenv import load_dotenv

# ============================================================
# ENV
# ============================================================
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("❌ HF_TOKEN missing in .env")

# ============================================================
# MODELS
# ============================================================
PRIMARY_MODEL  = "Qwen/Qwen2.5-7B-Instruct"
FALLBACK_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
HF_API_URL = "https://router.huggingface.co/v1/chat/completions"

# ============================================================
# HTTP SESSION
# ============================================================
_session = requests.Session()
_session.headers.update({
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json",
})

# ============================================================
# TOKEN UTILS
# ============================================================
def est_tokens(text: str) -> int:
    return max(1, len(text) // 3)

@lru_cache(maxsize=128)
def choose_max_tokens(question: str) -> int:
    if "short" in question.lower():
        return 150
    if len(question.split()) <= 5:
        return 200
    return 400

# ============================================================
# CONTEXT BUILDER
# ============================================================
def build_context(chunks: list[str], max_tokens: int = 900) -> str:
    if not chunks:
        return ""

    seen = set()
    used = 0
    out = []

    for i, c in enumerate(chunks, 1):
        c = c.strip()
        if not c or c in seen:
            continue
        seen.add(c)

        t = est_tokens(c)
        if used + t > max_tokens:
            break

        out.append(f"[DOC {i}]\n{c}")
        used += t

    return "\n\n".join(out)

# ============================================================
# RELEVANCE HEURISTIC (CRITICAL)
# ============================================================
def is_relevant(question: str, chunks: list[str], min_hits: int = 2) -> bool:
    q_words = set(question.lower().split())
    hits = 0

    for c in chunks:
        c_words = set(c.lower().split())
        if len(q_words & c_words) >= 1:
            hits += 1
        if hits >= min_hits:
            return True

    return False

# ============================================================
# HF ROUTER CALL
# ============================================================
def call_hf_api(messages: list[dict], model_id: str, max_tokens: int, rag: bool) -> str:
    payload = {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.2 if rag else 0.7,
        "top_p": 0.9,
        "frequency_penalty": 0.1,
        "provider": "auto",
    }

    try:
        r = _session.post(HF_API_URL, json=payload, timeout=45)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"].strip()
        if r.status_code == 503:
            time.sleep(8)
            return call_hf_api(messages, model_id, max_tokens, rag)
        return f"❌ API error {r.status_code}: {r.text[:120]}"
    except Exception as e:
        return f"❌ Request failed: {str(e)[:120]}"

# ============================================================
# MAIN ENTRY (AUTO RAG / LLM SWITCH)
# ============================================================
def ask_local(question: str, chunks: list[str] | None = None) -> str:
    if not question.strip():
        return "Please ask a question."

    chunks = chunks or []
    context = build_context(chunks)
    relevant = bool(context) and is_relevant(question, chunks)
    max_out = choose_max_tokens(question)

    # --------------------------------------------------------
    # RAG MODE (docs exist AND are relevant)
    # --------------------------------------------------------
    if relevant:
        system_prompt = f"""
You are a retrieval-augmented assistant.

RULES:
- Answer using ONLY the information in <context>.
- If the answer is not contained, say:
  "I don’t know based on the provided documents."
- Do NOT guess.

<context>
{context}
</context>
""".strip()
        rag = True

    # --------------------------------------------------------
    # NORMAL LLM MODE
    # --------------------------------------------------------
    else:
        system_prompt = "You are a helpful assistant. Answer naturally and concisely."
        rag = False

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]

    answer = call_hf_api(messages, PRIMARY_MODEL, max_out, rag)
    if not answer.startswith("❌"):
        return answer

    return call_hf_api(messages, FALLBACK_MODEL, max_out, rag)

# ============================================================
# ALIAS
# ============================================================
ask_gemini = ask_local
