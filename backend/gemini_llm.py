# backend/gemini_llm.py - answers via the Google Gemini API (remote, needs GEMINI_API_KEY)
import os
import time
import requests
from dotenv import load_dotenv

# ============================================================
# ENV (SAFE LOAD – NO CRASH ON IMPORT)
# ============================================================
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ============================================================
# MODELS
# "latest" aliases follow Google's current release, so retiring an
# old model version does not break the app.
# ============================================================
MODELS = [
    "gemini-flash-lite-latest",
    "gemini-flash-latest",
]
API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

# ============================================================
# HTTP SESSION (key goes in a header, never in the URL)
# ============================================================
_session = requests.Session()
_session.headers.update({"Content-Type": "application/json"})
if GEMINI_API_KEY:
    _session.headers.update({"x-goog-api-key": GEMINI_API_KEY})

# ============================================================
# TOKEN UTILS
# ============================================================
def est_tokens(text: str) -> int:
    return max(1, len(text) // 3)

def choose_max_tokens(question: str) -> int:
    q = question.lower()
    if "short" in q:
        return 150
    if len(question.split()) <= 5:
        return 200
    return 400

# ============================================================
# CONTEXT BUILDER (RAG)
# ============================================================
def select_chunks(chunks: list[str], max_tokens: int = 900) -> list[str]:
    """Deduplicate chunks and keep the ones that fit in the context budget."""
    if not chunks:
        return []

    seen = set()
    used = 0
    out = []

    for c in chunks:
        c = c.strip()
        if not c or c in seen:
            continue
        seen.add(c)

        t = est_tokens(c)
        if used + t > max_tokens:
            break

        out.append(c)
        used += t

    return out


def build_context(
    chunks: list[str], max_tokens: int = 900, files: list[str] | None = None
) -> str:
    """Label each excerpt with the file it came from, so the model can cite it
    and doesn't mistake an excerpt for a whole document."""
    source = {}
    for chunk, file in zip(chunks, files or []):
        source.setdefault(chunk.strip(), file)
    return "\n\n".join(
        f"[Excerpt {i} from {source.get(c) or 'an uploaded document'}]\n{c}"
        for i, c in enumerate(select_chunks(chunks, max_tokens), 1)
    )

# ============================================================
# RELEVANCE HEURISTIC
# ============================================================
STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "of", "in", "on", "to",
    "for", "and", "or", "it", "this", "that", "what", "which", "who", "how",
    "why", "when", "where", "do", "does", "did", "i", "you", "me", "my", "can",
    "about", "with", "from", "by", "as", "at", "document", "documents",
}

def keywords(text: str) -> set[str]:
    words = (w.strip(".,;:!?()[]\"'").lower() for w in text.split())
    return {w for w in words if len(w) > 2 and w not in STOPWORDS}

def is_relevant(question: str, chunks: list[str], min_hits: int = 1) -> bool:
    q_words = keywords(question)
    hits = 0

    for c in chunks:
        c_words = keywords(c)
        if len(q_words & c_words) >= 1:
            hits += 1
        if hits >= min_hits:
            return True

    return False


def retrieval_mode(question: str, chunks: list[str]) -> str:
    """"rag" when document chunks are sent to the model, "general" otherwise."""
    context = build_context(chunks)
    return "rag" if context and is_relevant(question, chunks) else "general"

# ============================================================
# GEMINI CALL
# ============================================================
def call_gemini(
    system_prompt: str,
    question: str,
    model: str,
    max_tokens: int,
    rag: bool,
    retries: int = 2,
) -> str:
    if not GEMINI_API_KEY:
        return "LLM not configured (GEMINI_API_KEY missing)."

    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": question}]}],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": 0.2 if rag else 0.7,
            "topP": 0.9,
        },
    }

    try:
        r = _session.post(f"{API_BASE}/{model}:generateContent", json=payload, timeout=45)

        if r.status_code == 200:
            data = r.json()
            parts = data["candidates"][0].get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts).strip()
            return text or "❌ Empty response from Gemini."

        # Overloaded or rate-limited: wait briefly and retry
        if r.status_code in (429, 503) and retries > 0:
            time.sleep(3)
            return call_gemini(system_prompt, question, model, max_tokens, rag, retries - 1)

        return f"❌ API error {r.status_code}: {r.text[:120]}"

    except Exception as e:
        return f"❌ Request failed: {str(e)[:120]}"

# ============================================================
# MAIN ENTRY (AUTO RAG / LLM)
# ============================================================
def ask_llm(
    question: str, chunks: list[str] | None = None, files: list[str] | None = None
) -> str:
    if not question.strip():
        return "Please ask a question."

    chunks = chunks or []
    context = build_context(chunks, files=files)
    relevant = bool(context) and is_relevant(question, chunks)
    max_out = choose_max_tokens(question)
    if relevant:
        system_prompt = f"""
You are a retrieval-augmented assistant.

The <context> holds excerpts retrieved from the user's uploaded documents,
not the whole documents. Each excerpt is labelled with its file name.

RULES:
- Use ONLY the information inside <context>.
- When you cite, name the file, not the excerpt number.
- If the answer is not present, say:
  "I don’t know based on the provided documents."
- Do NOT guess.

<context>
{context}
</context>
""".strip()
        rag = True
    else:
        system_prompt = "You are a helpful assistant. Answer naturally and concisely."
        rag = False

    # Try each model in order; fall back when one is busy or unavailable
    answer = ""
    for model in MODELS:
        answer = call_gemini(system_prompt, question, model, max_out, rag)
        if not answer.startswith("❌"):
            return answer

    return answer
