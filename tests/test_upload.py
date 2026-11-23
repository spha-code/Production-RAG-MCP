# tests/test_upload.py
import io

TEXT = b"Ibuprofen typical adult dose is 200-400 mg every 4 to 6 hours."


def upload(client, content=TEXT, name="ibuprofen.txt"):
    files = {"file": (name, io.BytesIO(content), "text/plain")}
    return client.post("/upload/", files=files, data={"chunk_size": 500, "chunk_overlap": 50})


def test_upload_text_file(client):
    response = upload(client)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["filename"] == "ibuprofen.txt"
    assert data["total_chunks"] == 1


def test_upload_is_searchable_immediately(client, mock_llm):
    """Uploaded chunks reach Chroma (with doc_id) without a restart."""
    upload(client)
    assert client.get("/health").json()["chroma_count"] == 4  # 3 seed + 1

    client.post("/chat", json={"query": "What is the ibuprofen dose?"})
    _, chunks = mock_llm.call_args.args
    assert any("Ibuprofen" in c for c in chunks)


def test_upload_clears_cache(client, mock_llm):
    client.post("/chat", json={"query": "What is the ibuprofen dose?"})
    upload(client)
    client.post("/chat", json={"query": "What is the ibuprofen dose?"})
    assert mock_llm.call_count == 2  # new document, so no stale cached answer


def test_list_and_delete_document(client):
    doc_id = upload(client).json()["document_id"]

    docs = client.get("/upload/documents").json()
    assert docs["total_documents"] == 1

    response = client.delete(f"/upload/documents/{doc_id}")
    assert response.status_code == 200
    assert client.get("/upload/documents").json()["total_documents"] == 0
    assert client.get("/health").json()["chroma_count"] == 3  # chunks removed too


def test_delete_unknown_document(client):
    assert client.delete("/upload/documents/does-not-exist").status_code == 404


def test_upload_unsupported_type(client):
    files = {"file": ("virus.exe", b"fake-executable-content", "application/octet-stream")}
    response = client.post("/upload/", files=files)
    assert response.status_code == 400
    assert "Unsupported extension" in response.json()["detail"]


def test_get_supported_types(client):
    response = client.get("/upload/supported-types")
    assert response.status_code == 200
    extensions = response.json()["supported_extensions"]
    assert ".pdf" in extensions and ".docx" in extensions
