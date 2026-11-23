# Production-RAG-MCP

A Retrieval-Augmented Generation (RAG) app: upload documents, then ask questions answered from them.
FastAPI backend, ChromaDB vector store, local Sentence-Transformers embeddings, Google Gemini for answers, and a small web chat UI.

## How it works

1. **Upload**: a file (PDF, DOCX, TXT, MD, CSV) is saved to `documents/`, split into ~1,000-character chunks (200 overlap), and each chunk is embedded locally with `all-MiniLM-L6-v2` (384 numbers per chunk) into Chroma (`chroma/`).
2. **Ask**: the question is embedded the same way and Chroma returns the closest chunks.
3. **Answer**: the chunks and the question go to Gemini, which answers only from them (or answers normally if no chunk is relevant).
4. **Cache**: answers are cached by meaning. A rephrased question (cosine distance ≤ 0.20) gets the saved answer without calling Gemini. The cache is cleared on startup and whenever documents change.

`chroma/` is generated: it's rebuilt from `documents/` on every startup and is not committed.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+.

```bash
uv sync
```

Create `.env` in the project root (it's git-ignored):

```
GEMINI_API_KEY=your_key_here
```

Get a free key at [aistudio.google.com](https://aistudio.google.com) → **Get API key**. The embedding model downloads automatically (~90 MB) on first start and needs no key.

## Run

**Backend only**, API docs at http://127.0.0.1:8000/docs:

```bash
uv run python -m backend.app
```

On Windows, set `PYTHONUTF8=1` first. The app prints emoji, which the default console encoding can't show:

```powershell
$env:PYTHONUTF8 = "1"; uv run python -m backend.app
```

**Full stack with the web UI** (backend + nginx), at http://localhost:

```bash
docker compose up -d --build
```

The web UI (`web/`) calls the backend through nginx at `/api/...`, so opening `web/index.html` directly won't work.

## API

| Method | Path | Description |
|---|---|---|
| `POST` | `/chat` | Ask a question: `{"query": "...", "k": 3}` → `{"answer": "..."}` |
| `POST` | `/upload/` | Upload a file (multipart: `file`, optional `metadata`, `chunk_size`, `chunk_overlap`) |
| `GET` | `/upload/documents` | List uploaded documents |
| `GET` | `/upload/documents/{id}` | One document's metadata |
| `DELETE` | `/upload/documents/{id}` | Delete a document and its chunks |
| `GET` | `/upload/supported-types` | Supported file extensions |
| `GET` | `/health` | Status and number of stored chunks |

## Project structure

```
backend/
  app.py              FastAPI app: startup indexing, /chat, /health
  gemini_llm.py       Gemini API calls and prompt building
  semantic_cache.py   Answer cache (separate Chroma collection)
  routes/upload.py    Upload, chunking, document management
documents/            Uploaded originals (source of truth)
chroma/               Vector DB, generated, git-ignored
web/                  Chat UI (HTML/JS/CSS)
deploy/               nginx configs (local and Docker)
tests/                pytest tests
test_docs/            Sample files to upload
```

## Known issues

- **Tests don't run.** `tests/conftest.py` imports `app` instead of `backend.app`, and some assertions expect old `/health` fields.
- **Not implemented yet** despite the name: an MCP endpoint and bulk folder upload.
