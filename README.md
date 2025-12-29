#### Run the repo:
``` uv run python backend/app.py```
http://127.0.0.1:8000/docs#/

```python -m http.server 8080```
http://localhost:8080/web/index.html


#### Clone empty repo from github:

```git clone https://github.com/spha-code/Production-RAG-MCP``` locally

```cd Production-RAG-MCP``` -->
```uv init``` -->
```uv add fastapi uvicorn sentence-transformers chromadb```

### Project Structure:

```
Production-RAG-MCP/
├── backend/                         # Python service + ML + MCP
│   ├── app.py                       # FastAPI entry: mounts routers
│   ├── config.py                    # env-based settings
│   ├── logging_config.py            # structured json logs
│   ├── telemetry.py                 # OTel + Prometheus
│   ├── exceptions.py                # custom HTTP handlers
│   ├── gunicorn_conf.py             # prod ASGI runner config
│   ├── routes/                      # API endpoints
│   │   ├── __init__.py
│   │   └── upload.py                # /upload endpoint
│   ├── middleware/
│   │   ├── __init__.py
│   │   └── auth.py                  # JWT / API-key guard
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── retriever.py             # Chroma & sentence-transformers
│   │   └── schemas.py               # Pydantic models
│   ├── mcp/
│   │   ├── __init__.py
│   │   └── tools.py                 # MCP tool descriptor + handler
│   ├── tests/
│   │   ├── __init__.py
│   │   └── test_app.py
│   ├── alembic/                     # DB migrations (if Postgres added)
│   │   └── versions/
│   ├── requirements-lock.txt
│   ├── requirements-dev.txt         # dev-only deps
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
└── .env                             # ignored
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

### demo.html 

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

### Add document upload - Multi-format file upload (PDF, DOCX, TXT, MD, CSV)

backend/routes/upload.py

```
# backend/routes/upload.py
import os
import csv
import json
import uuid
import hashlib
import shutil
import time
from pathlib import Path
from typing import List, Dict, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, File, Form, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import pandas as pd
from PyPDF2 import PdfReader
from docx import Document
import chardet

router = APIRouter(prefix="/upload", tags=["upload"])

# Supported file extensions
SUPPORTED_EXTENSIONS = {
    '.txt', '.md', '.pdf', '.docx', '.csv'
}

# Global progress tracker (in production, use Redis/database)
upload_progress: Dict[str, Any] = {}

def detect_file_encoding(file_path: str) -> str:
    """Detect file encoding for text files"""
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        return result['encoding'] or 'utf-8'

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF file"""
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF extraction failed: {str(e)}")

def extract_text_from_docx(file_path: str) -> str:
    """Extract text from DOCX file"""
    try:
        doc = Document(file_path)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"DOCX extraction failed: {str(e)}")

def extract_text_from_csv(file_path: str) -> str:
    """Extract text from CSV file - convert to readable format"""
    try:
        encoding = detect_file_encoding(file_path)
        df = pd.read_csv(file_path, encoding=encoding)
        
        text_parts = []
        text_parts.append(f"CSV with {len(df)} rows and {len(df.columns)} columns")
        text_parts.append("Columns: " + ", ".join(df.columns.tolist()))
        
        # Add first few rows as examples
        text_parts.append("\nSample data (first 3 rows):")
        for i, row in df.head(3).iterrows():
            row_text = " | ".join([f"{col}: {str(val)}" for col, val in row.items()])
            text_parts.append(f"Row {i+1}: {row_text}")
        
        # Add summary statistics if numeric columns exist
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            text_parts.append("\nNumeric column statistics:")
            for col in numeric_cols:
                mean_val = df[col].mean()
                max_val = df[col].max()
                min_val = df[col].min()
                text_parts.append(f"{col}: mean={mean_val:.2f}, range=[{min_val}, {max_val}]")
        
        return "\n".join(text_parts)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV extraction failed: {str(e)}")

def extract_text_from_txt(file_path: str) -> str:
    """Extract text from TXT/MD files"""
    try:
        encoding = detect_file_encoding(file_path)
        with open(file_path, 'r', encoding=encoding) as f:
            return f.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Text file reading failed: {str(e)}")

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """Split text into chunks with overlap"""
    if not text:
        return []
    
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        # Find the end of the chunk
        end = start + chunk_size
        
        # If this isn't the last chunk, try to break at a sentence
        if end < text_length:
            # Look for sentence boundaries
            sentence_end = text.rfind('.', end - 100, end)
            if sentence_end == -1:
                sentence_end = text.rfind('!', end - 100, end)
            if sentence_end == -1:
                sentence_end = text.rfind('?', end - 100, end)
            if sentence_end != -1:
                end = sentence_end + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position with overlap
        start = end - chunk_overlap
    
    return chunks

def process_document(file_path: str, filename: str, file_type: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> Dict[str, Any]:
    """Process document and extract text chunks"""
    
    # Extract text based on file type
    if file_type == 'pdf':
        text = extract_text_from_pdf(file_path)
    elif file_type == 'docx':
        text = extract_text_from_docx(file_path)
    elif file_type == 'csv':
        text = extract_text_from_csv(file_path)
    else:  # txt, md
        text = extract_text_from_txt(file_path)
    
    if not text:
        raise HTTPException(status_code=400, detail="No text content extracted")
    
    # Create chunks
    chunks = chunk_text(text, chunk_size, chunk_overlap)
    
    # Generate document metadata
    doc_id = str(uuid.uuid4())
    doc_hash = hashlib.md5(text.encode()).hexdigest()
    
    return {
        "document_id": doc_id,
        "filename": filename,
        "file_type": file_type,
        "text_hash": doc_hash,
        "total_chars": len(text),
        "total_chunks": len(chunks),
        "chunks": chunks
    }

@router.post("/")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    metadata: str = Form('{}'),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200)
):
    """Upload and process a single document"""
    
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    # Check file extension
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    
    # Create upload directory
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    
    # Save file temporarily
    file_path = os.path.join(upload_dir, f"{uuid.uuid4()}{file_ext}")
    
    try:
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process document
        file_type = file_ext.replace('.', '')
        result = process_document(file_path, file.filename, file_type, chunk_size, chunk_overlap)
        
        # Parse metadata
        try:
            custom_metadata = json.loads(metadata) if metadata else {}
        except json.JSONDecodeError:
            custom_metadata = {}
        
        # Store in vector database (background task)
        background_tasks.add_task(
            store_document_chunks,
            result["document_id"],
            result["chunks"],
            custom_metadata
        )
        
        return {
            "status": "success",
            "document_id": result["document_id"],
            "filename": result["filename"],
            "file_type": result["file_type"],
            "total_chunks": result["total_chunks"],
            "total_chars": result["total_chars"],
            "message": "Document uploaded and processing started"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
    
    finally:
        # Clean up temporary file
        if os.path.exists(file_path):
            os.remove(file_path)

@router.post("/folder")
async def upload_folder(
    background_tasks: BackgroundTasks,
    folder_path: str = Form(...),
    metadata: str = Form('{}'),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
    recursive: bool = Form(True),
    max_files: int = Form(100)
):
    """Upload and process all documents in a folder"""
    
    # Validate folder path
    folder = Path(folder_path)
    if not folder.exists():
        raise HTTPException(status_code=400, detail="Folder does not exist")
    if not folder.is_dir():
        raise HTTPException(status_code=400, detail="Path is not a directory")
    
    # Parse metadata
    try:
        custom_metadata = json.loads(metadata) if metadata else {}
    except json.JSONDecodeError:
        custom_metadata = {}
    
    # Scan for supported files
    files_to_process = []
    if recursive:
        # Walk through all subdirectories
        for file_path in folder.rglob('*'):
            if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                files_to_process.append(file_path)
                if len(files_to_process) >= max_files:
                    break
    else:
        # Only top-level files
        for file_path in folder.glob('*'):
            if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                files_to_process.append(file_path)
                if len(files_to_process) >= max_files:
                    break
    
    if not files_to_process:
        return {
            "status": "no_files",
            "message": f"No supported files found in folder. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        }
    
    # Process files in background
    background_tasks.add_task(
        process_folder_documents,
        files_to_process,
        custom_metadata,
        chunk_size,
        chunk_overlap
    )
    
    return {
        "status": "processing_started",
        "total_files": len(files_to_process),
        "folder_path": str(folder),
        "recursive": recursive,
        "message": f"Processing {len(files_to_process)} files in background"
    }

@router.post("/folder-with-progress")
async def upload_folder_with_progress(
    background_tasks: BackgroundTasks,
    folder_path: str = Form(...),
    metadata: str = Form('{}'),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
    recursive: bool = Form(True),
    max_files: int = Form(100)
):
    """Upload folder with progress tracking"""
    
    upload_id = str(uuid.uuid4())
    
    # Initialize progress
    upload_progress[upload_id] = {
        "status": "scanning",
        "total_files": 0,
        "processed_files": 0,
        "current_file": "",
        "errors": [],
        "start_time": time.time()
    }
    
    # Start background processing
    background_tasks.add_task(
        process_folder_with_progress,
        upload_id,
        folder_path,
        json.loads(metadata) if metadata else {},
        chunk_size,
        chunk_overlap,
        recursive,
        max_files
    )
    
    return {
        "upload_id": upload_id,
        "status": "started",
        "message": "Upload started. Check progress with GET /upload/progress/{upload_id}"
    }

@router.get("/progress/{upload_id}")
async def get_upload_progress(upload_id: str):
    """Get upload progress"""
    if upload_id not in upload_progress:
        raise HTTPException(status_code=404, detail="Upload ID not found")
    
    progress = upload_progress[upload_id]
    elapsed_time = time.time() - progress["start_time"]
    
    return {
        "upload_id": upload_id,
        "status": progress["status"],
        "total_files": progress["total_files"],
        "processed_files": progress["processed_files"],
        "current_file": progress["current_file"],
        "errors": progress["errors"],
        "elapsed_time": round(elapsed_time, 2),
        "completion_percentage": round(
            (progress["processed_files"] / max(progress["total_files"], 1)) * 100, 1
        )
    }

@router.get("/supported-types")
async def get_supported_types():
    """Get list of supported file types"""
    return {
        "supported_extensions": list(SUPPORTED_EXTENSIONS),
        "max_file_size": "50MB",  # You can adjust this
        "supported_formats": {
            "pdf": "Portable Document Format",
            "docx": "Microsoft Word Document",
            "txt": "Plain Text File",
            "md": "Markdown File",
            "csv": "Comma-Separated Values"
        }
    }

# Background processing functions
async def process_folder_documents(
    files: List[Path],
    metadata: Dict[str, Any],
    chunk_size: int,
    chunk_overlap: int
):
    """Process multiple documents from folder"""
    
    results = []
    total_chunks = 0
    
    # Use thread pool for concurrent processing
    with ThreadPoolExecutor(max_workers=4) as executor:
        loop = asyncio.get_event_loop()
        
        for file_path in files:
            try:
                # Process each file
                file_type = file_path.suffix.lower().replace('.', '')
                result = await loop.run_in_executor(
                    executor,
                    process_document_sync,
                    str(file_path),
                    file_path.name,
                    file_type,
                    chunk_size,
                    chunk_overlap
                )
                
                if result:
                    # Add folder metadata
                    result["folder_metadata"] = {
                        "relative_path": str(file_path.relative_to(file_path.parent.parent)) if len(file_path.parts) > 2 else file_path.name,
                        "file_size": file_path.stat().st_size,
                        "modified_time": file_path.stat().st_mtime
                    }
                    
                    results.append(result)
                    total_chunks += result["total_chunks"]
                    
            except Exception as e:
                results.append({
                    "filename": file_path.name,
                    "status": "failed",
                    "error": str(e)
                })
    
    # Store all chunks in vector database
    for result in results:
        if result.get("status") != "failed":
            await store_document_chunks_async(
                result["document_id"],
                result["chunks"],
                {**metadata, **result.get("folder_metadata", {})}
            )
    
    return {
        "processed_files": len(results),
        "total_chunks": total_chunks,
        "results": results
    }

def process_document_sync(
    file_path: str,
    filename: str,
    file_type: str,
    chunk_size: int,
    chunk_overlap: int
) -> Dict[str, Any]:
    """Synchronous version of document processing for thread pool"""
    
    if not os.path.exists(file_path):
        return None
    
    # Extract text based on file type
    if file_type == 'pdf':
        text = extract_text_from_pdf(file_path)
    elif file_type == 'docx':
        text = extract_text_from_docx(file_path)
    elif file_type == 'csv':
        text = extract_text_from_csv(file_path)
    else:  # txt, md
        text = extract_text_from_txt(file_path)
    
    if not text:
        return None
    
    # Create chunks
    chunks = chunk_text(text, chunk_size, chunk_overlap)
    
    # Generate document metadata
    doc_id = str(uuid.uuid4())
    doc_hash = hashlib.md5(text.encode()).hexdigest()
    
    return {
        "document_id": doc_id,
        "filename": filename,
        "file_type": file_type,
        "text_hash": doc_hash,
        "total_chars": len(text),
        "total_chunks": len(chunks),
        "chunks": chunks
    }

async def store_document_chunks_async(
    document_id: str, 
    chunks: List[str], 
    metadata: Dict[str, Any]
):
    """Async version to store chunks in vector database"""
    # This will connect to your existing ChromaDB setup
    # For now, it's a placeholder - implement based on your ChromaDB setup
    pass

async def store_document_chunks(
    document_id: str, 
    chunks: List[str], 
    metadata: Dict[str, Any]
):
    """Store chunks in vector database (background task)"""
    # This will connect to your existing ChromaDB setup
    # For now, it's a placeholder - implement based on your ChromaDB setup
    pass

async def process_folder_with_progress(
    upload_id: str,
    folder_path: str,
    metadata: Dict[str, Any],
    chunk_size: int,
    chunk_overlap: int,
    recursive: bool,
    max_files: int
):
    """Process folder with progress updates"""
    
    try:
        # Validate folder path
        folder = Path(folder_path)
        if not folder.exists() or not folder.is_dir():
            upload_progress[upload_id]["status"] = "failed"
            upload_progress[upload_id]["errors"].append("Invalid folder path")
            return
        
        # Scan for files
        files_to_process = []
        if recursive:
            for file_path in folder.rglob('*'):
                if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    files_to_process.append(file_path)
                    if len(files_to_process) >= max_files:
                        break
        else:
            for file_path in folder.glob('*'):
                if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    files_to_process.append(file_path)
                    if len(files_to_process) >= max_files:
                        break
        
        # Update progress
        upload_progress[upload_id]["total_files"] = len(files_to_process)
        upload_progress[upload_id]["status"] = "processing"
        
        if not files_to_process:
            upload_progress[upload_id]["status"] = "completed"
            return
        
        # Process files
        with ThreadPoolExecutor(max_workers=4) as executor:
            loop = asyncio.get_event_loop()
            
            for i, file_path in enumerate(files_to_process):
                try:
                    # Update current file
                    upload_progress[upload_id]["current_file"] = str(file_path)
                    
                    # Process file
                    file_type = file_path.suffix.lower().replace('.', '')
                    result = await loop.run_in_executor(
                        executor,
                        process_document_sync,
                        str(file_path),
                        file_path.name,
                        file_type,
                        chunk_size,
                        chunk_overlap
                    )
                    
                    if result:
                        # Store chunks
                        await store_document_chunks_async(
                            result["document_id"],
                            result["chunks"],
                            {**metadata, "source_file": str(file_path)}
                        )
                    
                    # Update progress
                    upload_progress[upload_id]["processed_files"] = i + 1
                    
                except Exception as e:
                    upload_progress[upload_id]["errors"].append(f"{file_path}: {str(e)}")
        
        # Mark as completed
        upload_progress[upload_id]["status"] = "completed"
        upload_progress[upload_id]["current_file"] = ""
        
    except Exception as e:
        upload_progress[upload_id]["status"] = "failed"
        upload_progress[upload_id]["errors"].append(str(e))
```

```cd backend```

```uv add PyPDF2 python-docx pandas chardet```


# Add to backend/app.py
```from .routes import upload```

# Added Testing at backend/tests

# Added CI/CD workflow

# Add router registration
```app.include_router(upload.router)```


