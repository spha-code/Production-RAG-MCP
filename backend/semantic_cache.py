# backend/semantic_cache.py - reuse answers for questions that mean the same thing
#
# Past questions are stored as embeddings in their own Chroma collection,
# next to (but separate from) the "docs" collection. Before calling the LLM,
# /chat looks for a past question close enough in meaning and returns its
# saved answer instead.
import uuid
from datetime import datetime, UTC

CACHE_COLLECTION = "answer_cache"

# Cosine distance: 0 = same meaning. Above this, questions count as different.
# Measured with all-MiniLM-L6-v2: rephrasings of one question scored 0.04-0.15,
# different questions ("capital of France" vs "of Germany", "dose" vs
# "side effects") scored 0.30+. 0.20 sits in the gap.
CACHE_MAX_DISTANCE = 0.20


def open_cache(chroma_client):
    return chroma_client.get_or_create_collection(
        name=CACHE_COLLECTION, metadata={"hnsw:space": "cosine"}
    )


def lookup(cache, embedding: list[list[float]]) -> str | None:
    """Return the saved answer for the closest past question, or None."""
    if cache.count() == 0:
        return None
    hit = cache.query(query_embeddings=embedding, n_results=1)
    if not hit["ids"][0]:
        return None
    distance = hit["distances"][0][0]
    if distance > CACHE_MAX_DISTANCE:
        return None
    print(f"⚡ Cache hit (distance {distance:.3f}): {hit['documents'][0][0]!r}")
    return hit["metadatas"][0][0]["answer"]


def save(cache, embedding: list[list[float]], question: str, answer: str) -> None:
    """Store a question and its answer. Errors are never cached."""
    if not answer or answer.startswith("❌") or "not configured" in answer:
        return
    cache.add(
        ids=[str(uuid.uuid4())],
        embeddings=embedding,
        documents=[question],
        metadatas=[{"answer": answer, "created": datetime.now(UTC).isoformat()}],
    )


def clear(app) -> None:
    """Drop all cached answers, e.g. after documents change."""
    client = app.state.chroma
    try:
        client.delete_collection(CACHE_COLLECTION)
    except Exception:
        pass
    app.state.cache = open_cache(client)
    print("🧹 Answer cache cleared")
