#### Clone empty repo from github:

```git clone https://github.com/spha-code/Production-RAG-MCP``` locally

```cd Production-RAG-MCP``` -->
```uv init``` -->
```uv add fastapi uvicorn sentence-transformers chromadb```

### Project Structure:

```
Production-RAG-MCP/
├── backend/                 # Python service + ML + MCP
│   ├── app.py               # FastAPI entry: /chat + /mcp routes
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── retriever.py     # Chroma & sentence-transformers
│   │   └── schemas.py       # Pydantic models
│   ├── mcp/
│   │   ├── __init__.py
│   │   └── tools.py         # MCP tool descriptor + handler
│   ├── tests/
│   │   ├── __init__.py
│   │   └── test_app.py
│   ├── requirements-lock.txt
│   └── Dockerfile
├── web/                     # Next.js site & embeddable widget
│   ├── pages/
│   ├── components/
│   │   └── ChatWidget.tsx
│   ├── public/
│   ├── styles/
│   └── package.json
├── ios/                     # SwiftUI Xcode project
│   └── ProductionRAG/
├── android/                 # Kotlin Android-Studio project
│   └── app/
├── iac/                     # AWS SAM infra
│   └── template.yaml
├── .github/
│   └── workflows/
│       └── ci-cd.yaml
├── scripts/
│   ├── build.sh
│   └── deploy.sh
├── pyproject.toml
├── uv.lock
└── README.md
```

On root directory:

```mkdir bakcend``` -->
```cd backend``` -->
```touch app.py```

### backend/app.py

```
# backend/app.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
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
```


```uv run python app.py```
```uv run python backend/app.py```

http://127.0.0.1:8000/docs

in root directory:

```mkdir web```
```touch web/demo.html```

```demo.html```

```
<!-- web/demo.html -->
<!doctype html>
<html>

<head>
    <meta charset="utf-8" />
    <title>ProductionRAG-MCP demo</title>
    <style>
        #rag-widget {
            position: fixed;
            bottom: 20px;
            right: 20px;
            width: 320px;
            height: 400px;
            border: 1px solid #ccc;
            border-radius: 8px;
            background: #fff;
            box-shadow: 0 4px 12px rgba(0, 0, 0, .15);
            display: none;
            flex-direction: column;
            padding: 12px;
        }

        #rag-widget.show {
            display: flex;
        }

        #rag-input {
            margin-top: auto;
        }

        #rag-log {
            flex: 1;
            overflow-y: auto;
            font-size: 14px;
        }

        button {
            cursor: pointer;
        }
    </style>
</head>

<body>
    <h1>Any dummy page content</h1>

    <!-- floating button -->
    <button id="toggle-btn" onclick="toggle()">💬</button>

    <!-- chat panel -->
    <div id="rag-widget">
        <div id="rag-log"></div>
        <input id="rag-input" placeholder="Ask anything…" />
        <button onclick="send()">Send</button>
    </div>

    <script>
        const API = 'http://127.0.0.1:8000/chat';
        function toggle() {
            document.getElementById('rag-widget').classList.toggle('show');
        }
        async function send() {
            const q = document.getElementById('rag-input').value;
            if (!q) return;
            const res = await fetch(API, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: q, k: 2 })
            }).then(r => r.json());
            document.getElementById('rag-log').innerText = res.chunks.join('\n');
        }
    </script>
</body>

</html>
```