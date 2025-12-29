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
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, File, Form, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
import pandas as pd
from PyPDF2 import PdfReader
from docx import Document
import chardet
import chromadb

router = APIRouter(tags=["upload"])

# Supported file extensions
SUPPORTED_EXTENSIONS = {
    '.txt', '.md', '.pdf', '.docx', '.csv'
}

# Production document storage
DOCUMENTS_DIR = "documents"
os.makedirs(DOCUMENTS_DIR, exist_ok=True)

# Document metadata storage (in production, use a proper database)
documents_metadata = {}

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
# temporary test
@router.get("/test")          
async def test_alive():
    return {"status": "ok"}

# Production document storage
@router.post("/")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    metadata: str = Form('{}'),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200)
):
    """Upload and process documents with production tracking"""
    
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Supported: {', '.join(SUPPORTED_EXTENSIONS)}")
    
    try:
        # Create document ID and folders
        doc_id = str(uuid.uuid4())
        doc_folder = os.path.join(DOCUMENTS_DIR, doc_id)
        os.makedirs(doc_folder, exist_ok=True)
        
        # Save original file
        original_path = os.path.join(doc_folder, f"original{file_ext}")
        with open(original_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process document
        file_type = file_ext.replace('.', '')
        result = process_document(original_path, file.filename, file_type, chunk_size, chunk_overlap)
        
        # Store metadata
        doc_metadata = {
            "document_id": doc_id,
            "filename": file.filename,
            "file_type": file_type,
            "file_size": len(content),
            "upload_time": datetime.now().isoformat(),
            "original_path": original_path,
            "total_chunks": result["total_chunks"],
            "text_hash": result["text_hash"],
            "custom_metadata": json.loads(metadata) if metadata else {}
        }
        
        documents_metadata[doc_id] = doc_metadata
        
        # Store in vector database
        background_tasks.add_task(
            store_document_chunks,
            doc_id,
            result["chunks"],
            doc_metadata
        )
        
        return {
            "status": "success",
            "document_id": doc_id,
            "filename": file.filename,
            "file_type": file_type,
            "total_chunks": result["total_chunks"],
            "file_size": len(content),
            "message": "Document uploaded and stored successfully",
            "document_url": f"/upload/documents/{doc_id}/download"
        }
        
    except Exception as e:
        # Clean up on error
        if 'doc_folder' in locals() and os.path.exists(doc_folder):
            shutil.rmtree(doc_folder)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

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

# Document Management Endpoints
@router.get("/documents")
async def list_documents():
    """List all uploaded documents with metadata"""
    return {
        "total_documents": len(documents_metadata),
        "documents": list(documents_metadata.values())
    }

@router.get("/documents/{document_id}")
async def get_document(document_id: str):
    """Get specific document details"""
    if document_id not in documents_metadata:
        raise HTTPException(status_code=404, detail="Document not found")
    return documents_metadata[document_id]

@router.get("/documents/{document_id}/download")
async def download_document(document_id: str):
    """Download original document file"""
    if document_id not in documents_metadata:
        raise HTTPException(status_code=404, detail="Document not found")
    
    doc_info = documents_metadata[document_id]
    file_path = doc_info["original_path"]
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Original file not found")
    
    return FileResponse(file_path, filename=doc_info["filename"])

@router.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete document and its data"""
    if document_id not in documents_metadata:
        raise HTTPException(status_code=404, detail="Document not found")
    
    try:
        doc_info = documents_metadata[document_id]
        
        # Delete document folder
        doc_folder = os.path.dirname(doc_info["original_path"])
        if os.path.exists(doc_folder):
            shutil.rmtree(doc_folder)
        
        # Remove from metadata
        del documents_metadata[document_id]
        
        return {"status": "success", "message": "Document deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")

@router.get("/supported-types")
async def get_supported_types():
    """Get list of supported file types"""
    return {
        "supported_extensions": list(SUPPORTED_EXTENSIONS),
        "max_file_size": "50MB",
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
    # This connects to your existing ChromaDB setup
    try:
        # For now, this is a placeholder - implement based on your setup
        # You'll need to access the app state to get the collection
        pass
    except Exception as e:
        print(f"Error storing chunks: {e}")

async def store_document_chunks(
    document_id: str, 
    chunks: List[str], 
    metadata: Dict[str, Any]
):
    """Store chunks in vector database (background task)"""
    # This connects to your existing ChromaDB setup
    try:
        # For now, this is a placeholder - implement based on your setup
        pass
    except Exception as e:
        print(f"Error storing chunks: {e}")