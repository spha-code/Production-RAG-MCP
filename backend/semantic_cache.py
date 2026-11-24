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


def save(
    cache,
    embedding: list[list[float]],
    question: str,
    answer: str,
    doc_ids=(),
) -> None:
    """Store a question and its answer, remembering which documents it used.

    Errors are never cached. `doc_ids` lets an edit or delete discard only the
    answers that actually relied on a document, instead of the whole cache.
    """
    if not answer or answer.startswith("❌") or "not configured" in answer:
        return
    sources = ",".join(sorted({d for d in doc_ids if d}))
    cache.add(
        ids=[str(uuid.uuid4())],
        embeddings=embedding,
        documents=[question],
        metadatas=[{
            "answer": answer,
            "created": datetime.now(UTC).isoformat(),
            "doc_ids": sources,
        }],
    )


def forget_documents(cache, doc_ids) -> int:
    """Delete cached answers that were built from any of `doc_ids`.

    Entries saved before source tracking existed have no "doc_ids" key. They
    can't be attributed to a document, so they are treated as affected and
    removed, which keeps stale answers out even for old cache entries.
    """
    doc_ids = {d for d in doc_ids if d}
    if not doc_ids or cache.count() == 0:
        return 0

    entries = cache.get(include=["metadatas"])
    stale = []
    for entry_id, meta in zip(entries["ids"], entries["metadatas"]):
        sources = (meta or {}).get("doc_ids")
        if sources is None or set(filter(None, sources.split(","))) & doc_ids:
            stale.append(entry_id)

    if stale:
        cache.delete(ids=stale)
        print(f"🧹 Forgot {len(stale)} cached answer(s) for {', '.join(sorted(doc_ids))}")
    return len(stale)


def clear(app) -> None:
    """Drop all cached answers, e.g. for a full reset."""
    client = app.state.chroma
    try:
        client.delete_collection(CACHE_COLLECTION)
    except Exception:
        pass
    app.state.cache = open_cache(client)
    print("🧹 Answer cache cleared")
