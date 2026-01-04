# backend/app.py
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer
from datetime import datetime
import pathlib
import json
from collections import defaultdict
from local_llm import ask_local as ask_gemini

from routes.upload import process_document_sync

thread_store = defaultdict(list)   # thread_id -> list[dict]

# ---------- import upload router ----------
print("=== DEBUG: Starting app setup ===")
try:
    from routes.upload import router as upload_router
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


# Track startup time
startup_time = time.time()

# ---------- lifespan: cold-start code ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🔄 Starting application lifespan...")
    
    # Store startup time
    app.state.start_time = time.time()
    
    # 1. Load SentenceTransformer
    app.state.encoder = SentenceTransformer("all-MiniLM-L6-v2")
    print("✅ SentenceTransformer loaded")
    
    # 2. Initialize ChromaDB
    app.state.chroma = chromadb.PersistentClient(path="./chroma")
    
    # 3. Get or create OPTIMIZED collection
    try:
        app.state.collection = app.state.chroma.get_collection("docs")
        print(f"📚 Loaded existing collection with {app.state.collection.count()} documents")
    except Exception:
        app.state.collection = app.state.chroma.create_collection(
            name="docs",
            metadata={"hnsw:space": "cosine"}  # OPTIMIZATION HERE
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
    
    # 5. Re-index previously uploaded documents
    from routes.upload import process_document_sync
    from routes.upload import documents_metadata
    
    docs_path = pathlib.Path("documents")
    if docs_path.exists():
        reindex_count = 0
        for doc_folder in docs_path.glob("*"):
            if not doc_folder.is_dir():
                continue
            
            try:
                original_file = None
                for ext in (".txt", ".md", ".pdf", ".docx", ".csv"):
                    candidate = doc_folder / f"original{ext}"
                    if candidate.exists():
                        original_file = candidate
                        break
                if not original_file:
                    continue

                file_type = original_file.suffix[1:]
                result = process_document_sync(
                    str(original_file),
                    original_file.name,
                    file_type,
                    chunk_size=1000,
                    chunk_overlap=200
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
                    
                    # Restore metadata
                    meta_file = doc_folder / "meta.json"
                    if meta_file.exists():
                        meta = json.loads(meta_file.read_text())
                    else:
                        meta = {
                            "document_id": doc_folder.name,
                            "filename": original_file.name,
                            "file_type": file_type,
                            "file_size": original_file.stat().st_size,
                            "upload_time": datetime.fromtimestamp(original_file.stat().st_mtime).isoformat(),
                            "original_path": str(original_file),
                            "total_chunks": len(result["chunks"]),
                            "text_hash": result["text_hash"],
                        }
                    documents_metadata[doc_folder.name] = meta
                    reindex_count += 1
                    
            except Exception as e:
                print(f"⚠️ Failed to re-index {doc_folder.name}: {e}")
        
        if reindex_count > 0:
            print(f"🔄 Re-indexed {reindex_count} documents")
    
    print("🚀 Application startup complete")
    yield   # FastAPI now serves requests


# ---------- create FastAPI instance ----------
app = FastAPI(title="ProductionRAG-MCP", version="0.1.0", lifespan=lifespan)

# ---------- CORS ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://127.0.0.1:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- include upload router ----------
if upload_router:
    app.include_router(upload_router, prefix="/upload", tags=["upload"])
    print("✅ Upload router included at /upload")
else:
    print("❌ Upload router not included")


# ---------- CHAT ENDPOINT ----------
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    print(f"\n=== CHAT REQUEST ===")
    print(f"Question: '{req.query}'")
    print(f"Thread: {req.thread_id}, k={req.k}")
    
    try:
        # 1. Retrieve relevant chunks from ChromaDB
        print("1. Retrieving chunks...")
        emb = app.state.encoder.encode([req.query]).tolist()
        res = app.state.collection.query(
            query_embeddings=emb,
            n_results=min(req.k, 5),
            where={"doc_id": {"$ne": "seed"}}
        )
        
        chunks = res["documents"][0] if res["documents"] else []
        print(f"   Found {len(chunks)} relevant chunks")
        
        # 2. Build conversation history
        history = thread_store[req.thread_id]
        conv = []
        for turn in history:
            conv.append(f"User: {turn['user']}")
            conv.append(f"Assistant: {turn['bot']}")
        conv.append(f"User: {req.query}")
        
        # 3. Generate answer using LLM
        print("2. Calling LLM...")
        llm_start = time.time()
        # Ensure ask_gemini is imported correctly
        answer = ask_gemini(req.query, chunks)
        llm_time = time.time() - llm_start
        print(f"   LLM response time: {llm_time:.2f}s")
        print(f"   Answer: '{answer[:100]}...'")
        
        # 4. Save conversation
        history.append({"user": req.query, "bot": answer})
        if len(history) > 20:
            thread_store[req.thread_id] = history[-20:]
        
        total_time = time.time() - app.state.start_time
        print(f"✅ Chat completed in {llm_time:.2f}s (total uptime: {total_time:.0f}s)")
        return ChatResponse(answer=answer)
        
    except Exception as e:
        print(f"❌ CHAT ERROR: {type(e).__name__}: {e}")
        return ChatResponse(answer="💬 I encountered an error. Please try again.")


# ---------- HEALTH CHECK ----------
@app.get("/health")
async def health_check():
    """System health check"""
    try:
        # Check if the LLM function is available instead of a global client variable
        # which was causing the import error in your tests.
        from local_llm import ask_local
        
        health_status = {
            "status": "healthy",
            "llm_loaded": ask_local is not None,
            "llm_type": "Gemini API",
            "chroma_count": app.state.collection.count() if hasattr(app.state, 'collection') else 0,
            "threads": len(thread_store),
            "uptime": time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0,
            "service": "ProductionRAG-MCP",
            "version": "0.1.0",
            "encoder_loaded": hasattr(app.state, 'encoder'),
        }
        
        return health_status
        
    except Exception as e:
        print(f"Health check error: {e}")
        return {
            "status": "degraded",
            "error": str(e),
            "service": "ProductionRAG-MCP"
        }


# ---------- DOCUMENT INFO ENDPOINT ----------
@app.get("/chat/documents/info")
async def get_document_info():
    """Return list of available documents"""
    try:
        from routes.upload import documents_metadata
        
        docs = []
        for doc_id, meta in documents_metadata.items():
            docs.append({
                "id": doc_id,
                "filename": meta.get("filename", "Unknown"),
                "file_type": meta.get("file_type", ""),
                "chunks": meta.get("total_chunks", 0),
                "upload_time": meta.get("upload_time", "")
            })
        
        return {"total_documents": len(docs), "documents": docs}
        
    except Exception as e:
        print(f"Document info error: {e}")
        return {"total_documents": 0, "documents": [], "error": str(e)}


# ---------- SIMPLE TEST ENDPOINT ----------
@app.get("/test")
async def test_endpoint():
    """Simple test endpoint"""
    return {"status": "ok", "message": "Backend is running", "timestamp": datetime.now().isoformat()}


# ---------- MCP tools descriptor ----------
@app.get("/mcp")
async def mcp_tools():
    return {
        "tools": [{
            "name": "semantic_search",
            "description": "Retrieve top-k relevant chunks for a query",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "k": {"type": "integer", "default": 3}
                },
                "required": ["query"]
            }
        }]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)