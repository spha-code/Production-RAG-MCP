# tests/test_upload.py
import io
import pytest

def test_upload_text_file(client):
    """Test uploading a simple text file"""
    # 1. Create a dummy text file in memory
    file_content = b"This is a test document about Paracetamol dosage. It should be chunked correctly."
    file_name = "test_meds.txt"
    
    # 2. Prepare the multipart form data
    files = {
        "file": (file_name, io.BytesIO(file_content), "text/plain")
    }
    data = {
        "chunk_size": 500,
        "chunk_overlap": 50
    }

    # 3. Post to the upload router (prefix is /upload in app.py)
    response = client.post("/upload/", files=files, data=data)

    # 4. Assertions
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "success"
    assert json_data["filename"] == file_name
    assert "document_id" in json_data
    assert json_data["total_chunks"] > 0

def test_list_documents(client):
    """Test the document management list endpoint"""
    response = client.get("/upload/documents")
    assert response.status_code == 200
    data = response.json()
    assert "total_documents" in data
    assert isinstance(data["documents"], list)

def test_upload_unsupported_type(client):
    """Test that unsupported files are rejected"""
    files = {
        "file": ("virus.exe", b"fake-executable-content", "application/octet-stream")
    }
    response = client.post("/upload/", files=files)
    assert response.status_code == 400
    assert "Unsupported extension" in response.json()["detail"]

def test_get_supported_types(client):
    """Test the metadata endpoint for supported types"""
    response = client.get("/upload/supported-types")
    assert response.status_code == 200
    assert ".pdf" in response.json()["supported_extensions"]
    assert ".docx" in response.json()["supported_extensions"]