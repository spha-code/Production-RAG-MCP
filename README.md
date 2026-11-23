#### Run the repo:
```uv run python -m backend.app``` ---> http://127.0.0.1:8000/docs#/

```sudo systemctl start nginx``` ---> http://localhost

Docker:

```docker compose up -d --build```

```docker compose up -d```


#### Clone empty repo from github:

```git clone https://github.com/spha-code/Production-RAG-MCP``` locally

```cd Production-RAG-MCP``` -->
```uv init``` -->
```uv add fastapi uvicorn sentence-transformers chromadb```

# Production-RAG-MCP

A production-ready **Retrieval-Augmented Generation (RAG)** system with FastAPI, ChromaDB, Sentence Transformers, and a lightweight web UI.  
Supports document ingestion (PDF, DOCX, TXT, CSV), semantic search, and MCP-compatible tool exposure.

---

## 🚀 Features

- FastAPI backend with lifecycle-managed embeddings
- ChromaDB vector store
- Sentence-Transformers embeddings
- File upload (PDF, DOCX, TXT, MD, CSV)
- Folder ingestion with progress tracking
- Chunking + metadata handling
- Simple web-based chat widget
- MCP-compatible `/mcp` endpoint
- Ready for MLOps, CI/CD, and scaling

---

## 📁 Project Structure

```
Production-RAG-MCP/
├── backend/                         # Python service + ML + MCP
│   ├── app.py                       # FastAPI entry: mounts routers
│   ├── routes/                      # API endpoints
│   │   ├── __init__.py
│   │   └── upload.py                # /upload endpoint
│   ├── middleware/
│   │   ├── __init__.py
│   │   └── auth.py                  # JWT / API-key guard
│   ├── rag/
│   │   ├── __init__.py
│   │   └── schemas.py               # Pydantic models
│   ├── mcp/
│   │   ├── __init__.py
│   │   └── tools.py                 # MCP tool descriptor + handler
│   ├── tests/
│   │   ├── __init__.py
│   │   └── test_app.py
│   ├── alembic/                     # DB migrations (if Postgres added)
│   │   └── versions/
│   ├── .env.example                 # template env vars
│   └── Dockerfile
├── mlops/                           # MLOps pipelines & monitoring
│   ├── data/                        # raw / labelled datasets
│   ├── notebooks/                   # EDA & embedding quality checks
│   ├── pipelines/
│   │   ├── embed_validation.py      # offline eval job
│   │   └── retrain_trigger.py       # scheduled / event retraining
│   ├── features/
│   │   └── chunker.py               # shared text-split logic
│   ├── monitoring/
│   │   ├── data_drift.py            # detect embedding drift
│   │   └── answer_quality.py        # LLM-as-judge scoring
│   ├── configs/
│   │   └── embed_config.yaml        # model, chunk size, overlap
│   ├── tests/
│   │   └── test_pipelines.py
│   ├── Dockerfile.pipeline          # Airflow / Prefect runner image
│   └── README.md                    # usage docs
├── web/                             # Next.js site & embeddable widget
│   ├── pages/
│   ├── components/
│   │   └── ChatWidget.tsx
│   ├── public/
│   ├── styles/
│   └── package.json
├── ios/                             # SwiftUI Xcode project
│   └── ProductionRAG/
├── android/                         # Kotlin Android-Studio project
│   └── app/
├── iac/                             # AWS SAM infra
│   └── template.yaml
├── .github/
│   └── workflows/
│       └── ci-cd.yaml
├── scripts/
│   ├── build.sh
│   └── deploy.sh
├── docker-compose.yml               # local dev stack
├── pyproject.toml
├── uv.lock
├── LICENSE
├── CHANGELOG.md
├── SECURITY.md
├── README.md
└── .env                             # .gitignore
```

### backend/app.py

### backend/routes/upload.py

### web/index.html 

### backend/local_llm.py

```uv add google-genai python-dotenv```

```cd backend ---> uv add PyPDF2 python-docx pandas chardet```

### API Endpoints

List all documents: http://127.0.0.1:8000/upload/documents

Test the API: http://127.0.0.1:8000/docs (Swagger UI)

Health check: http://127.0.0.1:8000/upload/test

| Method                   | Path                              | Description                               | Request                                                                            | Response                                                               |
| ------------------------ | --------------------------------- | ----------------------------------------- | ---------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| **Core RAG**             |                                   |                                           |                                                                                    |                                                                        |
| `POST`                   | `/chat`                           | Semantic search + generative answer       | `{ "query": "string", "k": 3 }`                                                    | `{ "chunks": ["concise answer"] }`                                     |
| `GET`                    | `/mcp`                            | MCP-compatible tool descriptor            | —                                                                                  | `{ "tools": [{ "name": "semantic_search", … }] }`                      |
| **Single-file Upload**   |                                   |                                           |                                                                                    |                                                                        |
| `POST`                   | `/upload/`                        | Upload one document (PDF/DOCX/TXT/MD/CSV) | multipart/form-data (`file`, `metadata`, `chunk_size`, `chunk_overlap`)            | `{ "document_id", "filename", "total_chunks", "message" }`             |
| **Bulk / Folder Upload** |                                   |                                           |                                                                                    |                                                                        |
| `POST`                   | `/upload/upload/folder`           | Queue an entire folder for ingestion      | `folder_path`, `metadata`, `chunk_size`, `chunk_overlap`, `recursive`, `max_files` | `{ "status": "processing_started", "total_files" }`                    |
| **Document Management**  |                                   |                                           |                                                                                    |                                                                        |
| `GET`                    | `/upload/documents`               | List every ingested document              | —                                                                                  | `{ "total_documents", "documents": [ … ] }`                            |
| `GET`                    | `/upload/documents/{id}`          | Get single document metadata              | —                                                                                  | `{ "document_id", "filename", "file_type", "upload_time", … }`         |
| `GET`                    | `/upload/documents/{id}/download` | Download original file                    | —                                                                                  | file stream (`Content-Disposition: attachment`)                        |
| `DELETE`                 | `/upload/documents/{id}`          | Delete document + chunks + file           | —                                                                                  | `{ "status": "success", "message" }`                                   |
| **Utility**              |                                   |                                           |                                                                                    |                                                                        |
| `GET`                    | `/upload/supported-types`         | Supported extensions & human names        | —                                                                                  | `{ "supported_extensions": [".pdf", …], "formats": { "pdf": "PDF" } }` |
| `GET`                    | `/upload/test`                    | Health ping                               | —                                                                                  | `{ "status": "ok" }`                                                   |
# Added Testing at backend/tests

# Added CI/CD workflow

# Add router registration
```app.include_router(upload.router)```

### Integrate llama LLM

```uv add llama-cpp-python```

Create file: ```backend/local_llm.py```

in app.py

# from gemini_client import ask_gemini
from local_llm import ask_local as ask_gemini   # same function signature

### Download the weights - fully open-source, offline RAG stack with no API bills.

```mkdir backend/models```
```uv run huggingface-cli login```
``` uv run hf download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf --local-dir models```







