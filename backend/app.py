# backend/app.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer

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

class ChatResponse(BaseModel):
    chunks: list[str]

# ---------- lifespan: cold-start code ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.encoder = SentenceTransformer("all-MiniLM-L6-v2")
    app.state.chroma  = chromadb.PersistentClient(path="./chroma")

    # 1. get or create collection
    try:
        app.state.collection = app.state.chroma.get_collection("docs")
    except Exception:
        app.state.collection = app.state.chroma.create_collection("docs")

    # 2. seed ONLY if collection is empty
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
            ids=[f"seed{i}" for i in range(len(docs))]
        )

    # 3. re-index anything previously uploaded to ./documents/
    import pathlib
    
    from routes.upload import process_document_sync

    docs_path = pathlib.Path("documents")
    for doc_folder in docs_path.glob("*"):
        if not doc_folder.is_dir():
            continue
        # find original file
        original_file = None
        for ext in (".txt", ".md", ".pdf", ".docx", ".csv"):
            candidate = doc_folder / f"original{ext}"
            if candidate.exists():
                original_file = candidate
                break
        if not original_file:
            continue

        file_type = original_file.suffix[1:]
        result      = process_document_sync(
            str(original_file),
            original_file.name,
            file_type,
            chunk_size=1000,
            chunk_overlap=200
        )
        if result and result["chunks"]:
            ids  = [f"{doc_folder.name}_{i}" for i in range(len(result["chunks"]))]
            embs = app.state.encoder.encode(result["chunks"]).tolist()
            app.state.collection.upsert(
                documents=result["chunks"],
                embeddings=embs,
                ids=ids,
                metadatas=[{"doc_id": doc_folder.name, "file": original_file.name}] * len(ids)
            )

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

# ---------- chat endpoint ----------
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    emb = app.state.encoder.encode([req.query]).tolist()
    res = app.state.collection.query(query_embeddings=emb, n_results=req.k)
    return ChatResponse(chunks=res["documents"][0])

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
                    "k":     {"type": "integer", "default": 3}
                },
                "required": ["query"]
            }
        }]
    }

# ---------- dev entry ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)