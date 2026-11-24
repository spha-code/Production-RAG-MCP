import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer
from datetime import datetime
import pathlib
import json
import re
from collections import defaultdict

# Module-style imports for Docker compatibility
from backend.gemini_llm import ask_llm, retrieval_mode, select_chunks
from backend import semantic_cache
from backend.routes.upload import process_document_sync, documents_metadata

thread_store = defaultdict(list)   # thread_id -> list[dict]

# Chunks further than this (cosine distance) are not sent to the model.
# Measured on the 2026 dyslipidemia guideline: real questions about it scored
# 0.30-0.37, unrelated ones ("capital of Japan", "hi", "do you have any
# documents?") 0.75+. Vector search always returns *something*; without this
# cutoff Gemini would be handed irrelevant excerpts and answer from them.
RETRIEVAL_MAX_DISTANCE = 0.60

# Questions about the collection itself ("which documents do you have?") are
# answered from the document list, not from RAG: the chunks can't know which
# files exist. Patterns are kept narrow so content questions like "which
# documents mention statins?" still go to RAG.
_DOC = r"(documents?|files?|docs|pdfs?)"
LIBRARY_PATTERNS = [
    rf"\b(do you have|are there|have you got) any {_DOC}\b",
    rf"\b(what|which) {_DOC} (do you have|have you got|are (there|stored|uploaded|available)|did i upload)\b",
    rf"\b(list|show)( me)?( all)?( (my|the|your))? {_DOC}\b",
    rf"\bhow many {_DOC}\b",
    r"\bwhat('s| is) in (my|your|the) (documents|library)\b",
]


def is_library_question(question: str) -> bool:
    q = question.lower()
    return any(re.search(p, q) for p in LIBRARY_PATTERNS)


def library_answer() -> str:
    docs = list(documents_metadata.values())
    if not docs:
        return "No documents are uploaded yet. Upload a file to ask questions about it."
    lines = [f"• {d.get('filename', '?')} ({d.get('total_chunks', '?')} chunks)" for d in docs]
    return f"{len(docs)} document(s) stored:\n" + "\n".join(lines)

# ---------- import upload router ----------
print("=== DEBUG: Starting app setup ===")
try:
    from backend.routes.upload import router as upload_router
    print(f"✅ Upload router imported: {upload_router}")
except Exception as e:
    print(f"❌ Import error: {e}")
    import traceback
    traceback.print_exc()
    upload_router = None

# ---------- request/response models ----------
class ChatRequest(BaseModel):
    query: str
    k: int = 3
    thread_id: str = "default"

class Source(BaseModel):
    doc_id: str
    file: str = ""
    chunk: str
    distance: float | None = None

class ChatResponse(BaseModel):
    answer: str
    mode: str = "general"          # "rag": chunks sent to the model; "library": answered from the document list
    sources: list[Source] = []     # the chunks the model was given, in order
    cached: bool = False           # answer replayed from the semantic cache

# ---------- lifespan: cold-start code ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🔄 Starting application lifespan...")
    
    app.state.start_time = time.time()
    
    # 1. Load SentenceTransformer
    app.state.encoder = SentenceTransformer("all-MiniLM-L6-v2")
    print("✅ SentenceTransformer loaded")
    
    # 2. Initialize ChromaDB
    # Absolute path: Chroma caches clients per path string, so a relative
    # "./chroma" would keep pointing at the first folder it was opened in.
    app.state.chroma = chromadb.PersistentClient(path=str(pathlib.Path("chroma").resolve()))
    
    # 3. Get or create collection
    try:
        app.state.collection = app.state.chroma.get_collection("docs")
        print(f"📚 Loaded existing collection with {app.state.collection.count()} documents")
    except Exception:
        app.state.collection = app.state.chroma.create_collection(
            name="docs",
            metadata={"hnsw:space": "cosine"}
        )
        print("📚 Created new optimized collection (hnsw:space=cosine)")
    
    # 4. Seed only if empty
    if app.state.collection.count() == 0:
        docs = [
            "GDPR applies to any company processing EU residents data.",
            "Paracetamol typical dose is 500-1000 mg.",
            "AWS free-tier includes 750 h of t2.micro per month."
        ]
        embs = app.state.encoder.encode(docs).tolist()
        app.state.collection.add(
            documents=docs,
            embeddings=embs,
            ids=[f"seed{i}" for i in range(len(docs))],
            metadatas=[{"doc_id": "seed", "file": "seed_data"}] * len(docs)
        )
        print(f"🌱 Seeded with {len(docs)} example documents")
    
    # 5. Re-index logic: only rewrite documents whose content actually changed.
    #    New or unchanged documents must not throw away cached answers; only an
    #    edited or deleted document can make an existing answer stale.
    docs_path = pathlib.Path("documents")
    changed_ids = set()  # edited or deleted documents: their answers are stale
    added_count = 0      # brand-new document; its chunks are added, answers stay valid
    if docs_path.exists():
        for doc_folder in docs_path.glob("*"):
            if not doc_folder.is_dir(): continue
            try:
                original_file = None
                for ext in (".txt", ".md", ".pdf", ".docx", ".csv"):
                    candidate = doc_folder / f"original{ext}"
                    if candidate.exists():
                        original_file = candidate
                        break
                if not original_file: continue

                file_type = original_file.suffix[1:]
                result = process_document_sync(
                    str(original_file), original_file.name, file_type,
                    chunk_size=1000, chunk_overlap=200
                )
                if not result or not result["chunks"]: continue

                meta_file = doc_folder / "meta.json"
                meta = json.loads(meta_file.read_text()) if meta_file.exists() else {
                    "document_id": doc_folder.name,
                    "filename": original_file.name,
                    "file_type": file_type,
                    "original_path": str(original_file),
                }

                # Same text as last time and its chunks are still indexed?
                # Then leave the document (and the answer cache) alone.
                stored = app.state.collection.get(where={"doc_id": doc_folder.name}, limit=1)
                same_text = meta.get("text_hash") == result["text_hash"]
                if same_text and stored["ids"]:
                    documents_metadata[doc_folder.name] = meta
                    continue

                # If we had a hash and it no longer matches, the content changed.
                # No hash and no chunks means this is simply a new document.
                content_changed = bool(meta.get("text_hash")) and not same_text

                # Content changed (or the index was lost): replace its chunks.
                if stored["ids"]:
                    app.state.collection.delete(where={"doc_id": doc_folder.name})
                ids = [f"{doc_folder.name}_{i}" for i in range(len(result["chunks"]))]
                embs = app.state.encoder.encode(result["chunks"]).tolist()
                app.state.collection.add(
                    documents=result["chunks"],
                    embeddings=embs,
                    ids=ids,
                    metadatas=[{"doc_id": doc_folder.name, "file": original_file.name}] * len(ids)
                )
                meta["text_hash"] = result["text_hash"]
                meta["total_chunks"] = len(result["chunks"])
                meta_file.write_text(json.dumps(meta, indent=2))
                documents_metadata[doc_folder.name] = meta
                if content_changed:
                    changed_ids.add(doc_folder.name)
                else:
                    added_count += 1
            except Exception as e:
                print(f"⚠️ Failed to re-index {doc_folder.name}: {e}")

        # Documents deleted outside the app: drop their leftover chunks.
        try:
            stored_docs = app.state.collection.get(
                where={"doc_id": {"$ne": "seed"}}, include=["metadatas"]
            )
            indexed_ids = {
                m.get("doc_id") for m in stored_docs["metadatas"] if m and m.get("doc_id")
            }
        except Exception:
            indexed_ids = set()
        on_disk = {p.name for p in docs_path.glob("*") if p.is_dir()}
        for orphan in indexed_ids - on_disk:
            try:
                app.state.collection.delete(where={"doc_id": orphan})
                changed_ids.add(orphan)
                print(f"🗑️ Removed chunks for deleted document {orphan}")
            except Exception as e:
                print(f"⚠️ Failed to remove chunks for {orphan}: {e}")

        if changed_ids:
            print(f"🔄 {len(changed_ids)} document change(s) detected: forgetting their answers")
        elif added_count:
            print(f"➕ {added_count} new document(s) indexed: keeping cached answers")
        else:
            print("✅ Documents unchanged")

    # 6. Answer cache: only discard answers that used an edited/deleted document.
    app.state.cache = semantic_cache.open_cache(app.state.chroma)
    if changed_ids:
        semantic_cache.forget_documents(app.state.cache, changed_ids)
    else:
        print(f"♻️ Keeping answer cache ({app.state.cache.count()} cached answers)")

    print("🚀 Application startup complete")
    yield

# ---------- create FastAPI instance ----------
app = FastAPI(title="ProductionRAG-MCP", version="0.1.0", lifespan=lifespan)

# ---------- MIDDLEWARE: Logging & CORS ----------

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = time.perf_counter() - start_time
    print(f"DEBUG: {request.method} {request.url.path} | Status: {response.status_code} | Time: {process_time:.4f}s")
    response.headers["X-Process-Time"] = str(process_time)
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# ---------- include routers ----------
if upload_router:
    app.include_router(upload_router, prefix="/upload", tags=["upload"])
    print("✅ Upload router included at /upload")

# ---------- ENDPOINTS ----------

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        # Questions about the collection itself: answer from the document list.
        # Not cached, so uploads and deletes show up immediately.
        if is_library_question(req.query):
            answer = library_answer()
            thread_store[req.thread_id].append({"user": req.query, "bot": answer})
            return ChatResponse(answer=answer, mode="library")

        # 1. Retrieve
        emb = app.state.encoder.encode([req.query]).tolist()

        # 0. Semantic cache: reuse the answer to a question with the same meaning
        cached = semantic_cache.lookup(app.state.cache, emb)
        if cached is not None:
            thread_store[req.thread_id].append({"user": req.query, "bot": cached["answer"]})
            return ChatResponse(
                answer=cached["answer"],
                mode=cached["mode"],
                sources=[Source(**s) for s in cached["sources"]],
                cached=True,
            )

        res = app.state.collection.query(
            query_embeddings=emb,
            n_results=min(req.k, 5),
            where={"doc_id": {"$ne": "seed"}},
            include=["documents", "metadatas", "distances"],
        )
        chunks = res["documents"][0] if res.get("documents") else []
        metas = res["metadatas"][0] if res.get("metadatas") else []
        distances = res["distances"][0] if res.get("distances") else []

        # Drop weak matches before anything else looks at them.
        kept = [
            (c, m, d) for c, m, d in zip(chunks, metas, distances)
            if d is not None and d <= RETRIEVAL_MAX_DISTANCE
        ]
        chunks = [c for c, _, _ in kept]
        metas = [m for _, m, _ in kept]
        distances = [d for _, _, d in kept]

        # Did document chunks actually reach the model? "general" means Gemini
        # answered from its own knowledge, so no embedding influenced the answer.
        mode = retrieval_mode(req.query, chunks)
        sent = set(select_chunks(chunks)) if mode == "rag" else set()

        sources: list[Source] = []
        for chunk, meta, distance in zip(chunks, metas, distances):
            if chunk.strip() not in sent:
                continue
            meta = meta or {}
            sources.append(Source(
                doc_id=str(meta.get("doc_id", "")),
                file=str(meta.get("file", "")),
                chunk=chunk[:300],
                distance=round(float(distance), 4) if distance is not None else None,
            ))

        # Which documents does this answer rest on? Stored with the cache entry
        # so editing/deleting a document only invalidates the answers that used it.
        doc_ids = {s.doc_id for s in sources if s.doc_id}

        # Level 1: server-side trace of what retrieval offered the model.
        if sources:
            detail = ", ".join(
                f"{s.file or s.doc_id} (distance {s.distance})" for s in sources
            )
            print(f"🔎 Retrieval: mode=rag, {len(sources)} chunk(s) sent: {detail}")
        else:
            print(
                f"🔎 Retrieval: mode={mode}, {len(chunks)} chunk(s) retrieved, "
                f"0 sent to the model"
            )

        # 2. History & LLM
        files = [str((m or {}).get("file", "")) for m in metas]
        answer = ask_llm(req.query, chunks, files=files)
        semantic_cache.save(
            app.state.cache, emb, req.query, answer, doc_ids, mode,
            [s.model_dump() for s in sources],
        )

        # 3. Save History
        thread_store[req.thread_id].append({"user": req.query, "bot": answer})
        if len(thread_store[req.thread_id]) > 20:
            thread_store[req.thread_id] = thread_store[req.thread_id][-20:]
            
        return ChatResponse(answer=answer, mode=mode, sources=sources)
    except Exception as e:
        print(f"❌ CHAT ERROR: {e}")
        return ChatResponse(answer="💬 I encountered an error. Please check backend logs.")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "chroma_count": app.state.collection.count() if hasattr(app.state, 'collection') else 0,
        "uptime": time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0
    }

@app.get("/chat/documents/info")
async def get_document_info():
    docs = [{"id": k, "filename": v.get("filename")} for k, v in documents_metadata.items()]
    return {"total_documents": len(docs), "documents": docs}

@app.get("/test")
async def test_endpoint():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)