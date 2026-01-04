import os
import json
import uuid
import hashlib
import shutil
import asyncio
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime, UTC

from fastapi import APIRouter, File, Form, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
import pandas as pd
from pypdf import PdfReader
from docx import Document
import chardet

router = APIRouter(tags=["upload"])

SUPPORTED_EXTENSIONS = {'.txt', '.md', '.pdf', '.docx', '.csv'}
DOCUMENTS_DIR = "documents"
os.makedirs(DOCUMENTS_DIR, exist_ok=True)

# In-memory registry (rebuilt from disk on startup in a real production app)
documents_metadata = {}

# ------------------------------------------------------------------
# Text Extractors
# ------------------------------------------------------------------
def detect_file_encoding(file_path: str) -> str:
    with open(file_path, 'rb') as f:
        return chardet.detect(f.read())['encoding'] or 'utf-8'

def extract_text_from_pdf(file_path: str) -> str:
    try:
        reader = PdfReader(file_path)
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    except Exception as e:
        raise HTTPException(400, f"PDF extraction failed: {e}")

def extract_text_from_docx(file_path: str) -> str:
    try:
        return "\n".join(p.text for p in Document(file_path).paragraphs)
    except Exception as e:
        raise HTTPException(400, f"DOCX extraction failed: {e}")

def extract_text_from_csv(file_path: str) -> str:
    encoding = detect_file_encoding(file_path)
    df = pd.read_csv(file_path, encoding=encoding)
    lines = [f"CSV with {len(df)} rows × {len(df.columns)} cols",
             "Columns: " + ", ".join(df.columns)]
    for i, row in df.head(3).iterrows():
        lines.append(f"Row {i+1}: " + " | ".join(f"{k}:{v}" for k, v in row.items()))
    return "\n".join(lines)

def extract_text_from_txt(file_path: str) -> str:
    with open(file_path, encoding=detect_file_encoding(file_path)) as f:
        return f.read()

# ------------------------------------------------------------------
# Chunking Logic
# ------------------------------------------------------------------
def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            for sep in ('. ', '! ', '? '):
                pos = text.rfind(sep, end - 100, end)
                if pos != -1:
                    end = pos + 2
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - chunk_overlap
    return chunks

# ------------------------------------------------------------------
# Core Processor
# ------------------------------------------------------------------
def process_document(file_path: str, filename: str, file_type: str,
                     chunk_size: int = 1000, chunk_overlap: int = 200) -> Dict[str, Any]:
    extract_map = {
        'pdf':  extract_text_from_pdf,
        'docx': extract_text_from_docx,
        'csv':  extract_text_from_csv,
        'txt':  extract_text_from_txt,
        'md':   extract_text_from_txt
    }
    
    if file_type not in extract_map:
        raise HTTPException(400, f"No extractor for {file_type}")
        
    text = extract_map[file_type](file_path)
    if not text:
        raise HTTPException(400, "No text extracted from file")
        
    chunks = chunk_text(text, chunk_size, chunk_overlap)
    return {
        "document_id": str(uuid.uuid4()),
        "filename": filename,
        "file_type": file_type,
        "text_hash": hashlib.md5(text.encode()).hexdigest(),
        "total_chars": len(text),
        "total_chunks": len(chunks),
        "chunks": chunks
    }

# ------------------------------------------------------------------
# ChromaDB Storage Helpers
# ------------------------------------------------------------------
def store_document_chunks(document_id: str, chunks: List[str], metadata: Dict[str, Any]):
    """
    Background task to embed and store chunks.
    """
    from app import app as fastapi_app 
    
    # 1. Prepare flat metadata for ChromaDB
    # Extract custom_metadata and remove it from the dict to avoid nesting
    custom_meta = metadata.pop("custom_metadata", {})
    
    # Merge them into a single flat dict
    flat_metadata = {**metadata, **custom_meta}
    
    ids = [f"{document_id}_{i}" for i in range(len(chunks))]
    embs = fastapi_app.state.encoder.encode(chunks).tolist()
    
    # 2. Use the flattened metadata
    fastapi_app.state.collection.upsert(
        documents=chunks,
        embeddings=embs,
        ids=ids,
        metadatas=[{**flat_metadata, "chunk_idx": i} for i in range(len(chunks))]
    )
# ------------------------------------------------------------------
# Background & Startup Helpers
# ------------------------------------------------------------------
def process_document_sync(
    file_path: str,
    filename: str,
    file_type: str,
    chunk_size: int,
    chunk_overlap: int
) -> Dict[str, Any]:
    """
    Wrapper for process_document used by app.py during startup re-indexing.
    """
    return process_document(file_path, filename, file_type, chunk_size, chunk_overlap)
# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------
@router.post("/")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    metadata: str = Form('{}'),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200)
):
    if not file.filename:
        raise HTTPException(400, "No filename provided")
        
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported extension. Allowed: {SUPPORTED_EXTENSIONS}")

    doc_id = str(uuid.uuid4())
    doc_folder = Path(DOCUMENTS_DIR) / doc_id
    doc_folder.mkdir(parents=True, exist_ok=True)
    original_path = doc_folder / f"original{ext}"

    content = await file.read()
    original_path.write_bytes(content)

    file_type = ext[1:] # remove the dot
    result = process_document(str(original_path), file.filename, file_type, chunk_size, chunk_overlap)

    doc_meta = {
        "document_id": doc_id,
        "filename": file.filename,
        "file_type": file_type,
        "file_size": len(content),
        "upload_time": datetime.now(UTC).isoformat(), # Modernized UTC call
        "original_path": str(original_path),
        "total_chunks": result["total_chunks"],
        "text_hash": result["text_hash"],
        "custom_metadata": json.loads(metadata) if metadata else {}
    }
    
    documents_metadata[doc_id] = doc_meta
    (doc_folder / "meta.json").write_text(json.dumps(doc_meta, indent=2))

    # Trigger indexing in the background
    background_tasks.add_task(store_document_chunks, doc_id, result["chunks"], doc_meta)

    return {
        "status": "success",
        "document_id": doc_id,
        "filename": file.filename,
        "file_type": file_type,
        "total_chunks": result["total_chunks"],
        "message": "Document uploaded and indexing started",
        "document_url": f"/upload/documents/{doc_id}/download"
    }

@router.get("/documents")
def list_documents():
    return {"total_documents": len(documents_metadata), "documents": list(documents_metadata.values())}

@router.get("/documents/{document_id}")
def get_document(document_id: str):
    if document_id not in documents_metadata:
        raise HTTPException(404, "Document not found")
    return documents_metadata[document_id]

@router.delete("/documents/{document_id}")
def delete_document(document_id: str):
    if document_id not in documents_metadata:
        raise HTTPException(404, "Document not found")
        
    doc_info = documents_metadata[document_id]
    folder = Path(doc_info["original_path"]).parent
    
    if folder.exists():
        shutil.rmtree(folder)
        
    del documents_metadata[document_id]
    return {"status": "success", "message": "Document deleted from disk"}

@router.get("/supported-types")
def supported_types():
    return {
        "supported_extensions": list(SUPPORTED_EXTENSIONS),
        "formats": {e[1:]: e.upper() for e in SUPPORTED_EXTENSIONS}
    }