# backend/routes/upload.py
import os
import csv
import json
import uuid
import hashlib
import shutil
import asyncio
import time
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, File, Form, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
import pandas as pd
from PyPDF2 import PdfReader
from docx import Document
import chardet

router = APIRouter(tags=["upload"])

SUPPORTED_EXTENSIONS = {'.txt', '.md', '.pdf', '.docx', '.csv'}

DOCUMENTS_DIR = "documents"
os.makedirs(DOCUMENTS_DIR, exist_ok=True)

documents_metadata = {}          # in-mem registry (rebuilt from disk on start)

# ------------------------------------------------------------------
# text extractors
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
# chunking
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
# core processor
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
    text = extract_map[file_type](file_path)
    if not text:
        raise HTTPException(400, "No text extracted")
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
# Chroma helpers
# ------------------------------------------------------------------
def store_document_chunks(document_id: str, chunks: List[str], metadata: Dict[str, Any]):
    from ..app import app
    ids  = [f"{document_id}_{i}" for i in range(len(chunks))]
    embs = app.state.encoder.encode(chunks).tolist()
    app.state.collection.upsert(
        documents=chunks,
        embeddings=embs,
        ids=ids,
        metadatas=[metadata | {"chunk_idx": i} for i in range(len(chunks))]
    )

async def store_document_chunks_async(document_id: str, chunks: List[str], metadata: Dict[str, Any]):
    store_document_chunks(document_id, chunks, metadata)

# ------------------------------------------------------------------
# single file upload
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
        raise HTTPException(400, "No filename")
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported extension. Allowed: {SUPPORTED_EXTENSIONS}")

    doc_id = str(uuid.uuid4())
    doc_folder = Path(DOCUMENTS_DIR) / doc_id
    doc_folder.mkdir(parents=True, exist_ok=True)
    original_path = doc_folder / f"original{ext}"

    content = await file.read()
    original_path.write_bytes(content)

    file_type = ext[1:]        # remove dot
    result = process_document(str(original_path), file.filename, file_type, chunk_size, chunk_overlap)

    # build metadata object
    doc_meta = {
        "document_id": doc_id,
        "filename": file.filename,          # real human name
        "file_type": file_type,
        "file_size": len(content),
        "upload_time": datetime.utcnow().isoformat(),
        "original_path": str(original_path),
        "total_chunks": result["total_chunks"],
        "text_hash": result["text_hash"],
        "custom_metadata": json.loads(metadata) if metadata else {}
    }
    documents_metadata[doc_id] = doc_meta

    # save small side-car file for next restart
    (doc_folder / "meta.json").write_text(json.dumps(doc_meta, indent=2))

    background_tasks.add_task(store_document_chunks, doc_id, result["chunks"], doc_meta)

    return {
        "status": "success",
        "document_id": doc_id,
        "filename": file.filename,
        "file_type": file_type,
        "total_chunks": result["total_chunks"],
        "file_size": len(content),
        "message": "Document uploaded and indexed",
        "document_url": f"/upload/documents/{doc_id}/download"
    }

# ------------------------------------------------------------------
# folder upload
# ------------------------------------------------------------------
@router.post("/upload/folder")
async def upload_folder(
    background_tasks: BackgroundTasks,
    folder_path: str = Form(...),
    metadata: str = Form('{}'),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
    recursive: bool = Form(True),
    max_files: int = Form(100)
):
    folder = Path(folder_path).expanduser().resolve()
    if not folder.is_dir():
        raise HTTPException(400, "Folder does not exist or is not a directory")

    files = list(folder.rglob("*")) if recursive else list(folder.glob("*"))
    files = [f for f in files if f.suffix.lower() in SUPPORTED_EXTENSIONS][:max_files]
    if not files:
        return {"status": "no_files", "message": "No supported files found"}

    meta = json.loads(metadata) if metadata else {}
    background_tasks.add_task(process_folder_documents, files, meta, chunk_size, chunk_overlap)
    return {"status": "processing_started", "total_files": len(files)}

# ------------------------------------------------------------------
# document management
# ------------------------------------------------------------------
@router.get("/documents")
def list_documents():
    return {"total_documents": len(documents_metadata), "documents": list(documents_metadata.values())}

@router.get("/documents/{document_id}")
def get_document(document_id: str):
    if document_id not in documents_metadata:
        raise HTTPException(404, "Document not found")
    return documents_metadata[document_id]

@router.get("/documents/{document_id}/download")
def download_document(document_id: str):
    if document_id not in documents_metadata:
        raise HTTPException(404, "Document not found")
    path = Path(documents_metadata[document_id]["original_path"])
    if not path.exists():
        raise HTTPException(404, "Original file gone")
    return FileResponse(path, filename=path.name)

@router.delete("/documents/{document_id}")
def delete_document(document_id: str):
    if document_id not in documents_metadata:
        raise HTTPException(404, "Document not found")
    doc_info = documents_metadata[document_id]
    folder = Path(doc_info["original_path"]).parent
    if folder.exists():
        shutil.rmtree(folder)
    del documents_metadata[document_id]
    return {"status": "success", "message": "Document deleted"}

@router.get("/supported-types")
def supported_types():
    return {
        "supported_extensions": list(SUPPORTED_EXTENSIONS),
        "formats": {e[1:]: e.upper() for e in SUPPORTED_EXTENSIONS}
    }

# ------------------------------------------------------------------
# background helpers
# ------------------------------------------------------------------
def process_document_sync(
    file_path: str,
    filename: str,
    file_type: str,
    chunk_size: int,
    chunk_overlap: int
) -> Dict[str, Any]:
    return process_document(file_path, filename, file_type, chunk_size, chunk_overlap)

async def process_folder_documents(
    files: List[Path],
    metadata: Dict[str, Any],
    chunk_size: int,
    chunk_overlap: int
):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=4) as pool:
        tasks = [
            loop.run_in_executor(pool, process_document_sync,
                                 str(f), f.name, f.suffix[1:].lower(),
                                 chunk_size, chunk_overlap)
            for f in files
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    for res in results:
        if isinstance(res, Exception):
            print("Folder processing error:", res)
            continue
        if res and res["chunks"]:
            store_document_chunks(res["document_id"], res["chunks"],
                                  metadata | res.get("folder_metadata", {}))

# ------------------------------------------------------------------
# health ping
# ------------------------------------------------------------------
@router.get("/test")
async def test_alive():
    return {"status": "ok"}