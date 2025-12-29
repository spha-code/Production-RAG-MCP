# backend/app.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer

# DEBUG: Import with error checking
print("=== DEBUG: Starting app setup ===")
try:
    from routes.upload import router as upload_router
    print(f"✅ Upload router imported: {upload_router}")
    print(f"✅ Router routes: {[route.path for route in upload_router.routes]}")
except Exception as e:
    print(f"❌ Import error: {e}")
    import traceback
    traceback.print_exc()
    upload_router = None

class ChatRequest(BaseModel):
    query: str
    k: int = 3

class ChatResponse(BaseModel):
    chunks: list[str]



@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.encoder = SentenceTransformer("all-MiniLM-L6-v2")
    app.state.chroma = chromadb.PersistentClient(path="./chroma")
    try:
        app.state.collection = app.state.chroma.get_collection("docs")
    except Exception:
        app.state.collection = app.state.chroma.create_collection("docs")
        docs = [
            "GDPR applies to any company processing EU residents data.",
            "Paracetamol typical dose is 500-1000 mg.",
            "AWS free-tier includes 750 h of t2.micro per month."
        ]
        embs = app.state.encoder.encode(docs).tolist()
        app.state.collection.add(documents=docs, ids=[f"id{i}" for i in range(len(docs))], embeddings=embs)
    yield

app = FastAPI(title="ProductionRAG-MCP", version="0.1.0", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://127.0.0.1:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include upload router only if it imported successfully
if upload_router:
    app.include_router(upload_router, prefix="/upload", tags=["upload"])
    print(f"✅ Upload router included with prefix: /upload")
else:
    print("❌ Upload router not included due to import error")

# DEBUG: Check all routes
print(f"✅ All app routes: {[route.path for route in app.routes]}")

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    emb = app.state.encoder.encode([req.query]).tolist()
    res = app.state.collection.query(query_embeddings=emb, n_results=req.k)
    return ChatResponse(chunks=res["documents"][0])

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)