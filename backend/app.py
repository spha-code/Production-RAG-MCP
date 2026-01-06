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
from collections import defaultdict

# Module-style imports for Docker compatibility
from backend.local_llm import ask_local as ask_gemini
from backend.routes.upload import process_document_sync, documents_metadata

thread_store = defaultdict(list)   # thread_id -> list[dict]

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

class ChatResponse(BaseModel):
    answer: str

# ---------- lifespan: cold-start code ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🔄 Starting application lifespan...")
    
    app.state.start_time = time.time()
    
    # 1. Load SentenceTransformer
    app.state.encoder = SentenceTransformer("all-MiniLM-L6-v2")
    print("✅ SentenceTransformer loaded")
    
    # 2. Initialize ChromaDB
    app.state.chroma = chromadb.PersistentClient(path="./chroma")
    
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
    
    # 5. Re-index logic
    docs_path = pathlib.Path("documents")
    if docs_path.exists():
        reindex_count = 0
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
                
                if result and result["chunks"]:
                    ids = [f"{doc_folder.name}_{i}" for i in range(len(result["chunks"]))]
                    embs = app.state.encoder.encode(result["chunks"]).tolist()
                    app.state.collection.upsert(
                        documents=result["chunks"],
                        embeddings=embs,
                        ids=ids,
                        metadatas=[{"doc_id": doc_folder.name, "file": original_file.name}] * len(ids)
                    )
                    
                    meta_file = doc_folder / "meta.json"
                    meta = json.loads(meta_file.read_text()) if meta_file.exists() else {
                        "document_id": doc_folder.name,
                        "filename": original_file.name,
                        "total_chunks": len(result["chunks"]),
                    }
                    documents_metadata[doc_folder.name] = meta
                    reindex_count += 1
            except Exception as e:
                print(f"⚠️ Failed to re-index {doc_folder.name}: {e}")
        if reindex_count > 0: print(f"🔄 Re-indexed {reindex_count} documents")
    
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
        # 1. Retrieve
        emb = app.state.encoder.encode([req.query]).tolist()
        res = app.state.collection.query(
            query_embeddings=emb,
            n_results=min(req.k, 5),
            where={"doc_id": {"$ne": "seed"}}
        )
        chunks = res["documents"][0] if res["documents"] else []
        
        # 2. History & LLM
        answer = ask_gemini(req.query, chunks)
        
        # 3. Save History
        thread_store[req.thread_id].append({"user": req.query, "bot": answer})
        if len(thread_store[req.thread_id]) > 20:
            thread_store[req.thread_id] = thread_store[req.thread_id][-20:]
            
        return ChatResponse(answer=answer)
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