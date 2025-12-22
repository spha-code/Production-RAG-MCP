# backend/app.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware 
from pydantic import BaseModel   # <-- validates JSON automatically
import chromadb
from sentence_transformers import SentenceTransformer


class ChatRequest(BaseModel):
    query: str
    k: int = 3          # top-k chunks to return

class ChatResponse(BaseModel):
    chunks: list[str]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1× cold-start code
    app.state.encoder = SentenceTransformer("all-MiniLM-L6-v2")
    app.state.chroma = chromadb.PersistentClient(path="./chroma")
    try:
        app.state.collection = app.state.chroma.get_collection("docs")
    except Exception:
        app.state.collection = app.state.chroma.create_collection("docs")
        # seed 3 tiny sentences so the demo never returns empty
        docs = [
            "GDPR applies to any company processing EU residents data.",
            "Paracetamol typical dose is 500-1000 mg.",
            "AWS free-tier includes 750 h of t2.micro per month."
        ]
        embs = app.state.encoder.encode(docs).tolist()
        app.state.collection.add(documents=docs, ids=[f"id{i}" for i in range(len(docs))], embeddings=embs)
    yield   # hands control to FastAPI
    # (anything after yield runs on shutdown – we don’t need it yet)


app = FastAPI(title="ProductionRAG-MCP", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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