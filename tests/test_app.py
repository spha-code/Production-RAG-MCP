# tests/test_app.py
import io

from fastapi.testclient import TestClient

from backend.app import app, thread_store
from backend.routes.upload import documents_metadata


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    # A fresh store holds only the 3 seed sentences
    assert data["chroma_count"] == 3
    assert data["uptime"] >= 0


def test_test_endpoint(client):
    response = client.get("/test")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "timestamp" in data


def test_chat_endpoint(client, mock_llm):
    payload = {"query": "What is the dose of Paracetamol?", "thread_id": "t1", "k": 3}
    response = client.post("/chat", json=payload)

    assert response.status_code == 200
    assert response.json()["answer"] == "Mocked Gemini Response"
    mock_llm.assert_called_once()


def test_chat_excludes_seed_data(client, mock_llm):
    """Seed sentences must never be sent to the LLM as context."""
    client.post("/chat", json={"query": "What is GDPR?"})
    _, chunks = mock_llm.call_args.args
    assert chunks == []


def test_chat_cache_reuses_answer_for_rephrased_question(client, mock_llm):
    first = client.post("/chat", json={"query": "What is the capital of France?"})
    second = client.post("/chat", json={"query": "What's the capital city of France?"})

    assert first.json()["answer"] == second.json()["answer"] == "Mocked Gemini Response"
    mock_llm.assert_called_once()  # the second answer came from the cache


def test_chat_cache_misses_for_different_question(client, mock_llm):
    client.post("/chat", json={"query": "What is the capital of France?"})
    client.post("/chat", json={"query": "What is the capital of Germany?"})
    assert mock_llm.call_count == 2


def test_chat_errors_are_not_cached(client, mock_llm):
    mock_llm.return_value = "❌ API error 503: busy"
    client.post("/chat", json={"query": "What is the capital of France?"})
    client.post("/chat", json={"query": "What is the capital of France?"})
    assert mock_llm.call_count == 2


def test_chat_reports_sources_when_documents_are_used(client, mock_llm):
    client.post(
        "/upload/",
        files={
            "file": (
                "ibuprofen.txt",
                io.BytesIO(b"Ibuprofen adult dose is 200-400 mg every 4 to 6 hours."),
                "text/plain",
            )
        },
        data={"chunk_size": 500, "chunk_overlap": 50},
    )
    data = client.post("/chat", json={"query": "What is the ibuprofen dose?", "k": 1}).json()

    assert data["mode"] == "rag"
    assert data["cached"] is False
    assert len(data["sources"]) == 1
    source = data["sources"][0]
    assert source["file"] == "ibuprofen.txt"
    assert "Ibuprofen" in source["chunk"]
    assert isinstance(source["distance"], float)


def test_chat_reports_general_mode_without_documents(client, mock_llm):
    data = client.post("/chat", json={"query": "What is the capital of France?"}).json()

    assert data["mode"] == "general"
    assert data["sources"] == []
    assert data["cached"] is False


def test_cache_hit_replays_mode_and_sources(client, mock_llm):
    client.post(
        "/upload/",
        files={
            "file": (
                "ibuprofen.txt",
                io.BytesIO(b"Ibuprofen adult dose is 200-400 mg every 4 to 6 hours."),
                "text/plain",
            )
        },
        data={"chunk_size": 500, "chunk_overlap": 50},
    )
    question = "What is the ibuprofen dose?"
    first = client.post("/chat", json={"query": question, "k": 1}).json()
    second = client.post("/chat", json={"query": question, "k": 1}).json()

    assert second["cached"] is True
    assert second["mode"] == first["mode"] == "rag"
    assert second["sources"] == first["sources"]
    mock_llm.assert_called_once()


def test_document_info_empty(client):
    response = client.get("/chat/documents/info")
    assert response.status_code == 200
    assert response.json() == {"total_documents": 0, "documents": []}


def test_chat_validation_error(client):
    response = client.post("/chat", json={})
    assert response.status_code == 422


def test_restart_keeps_cache_when_documents_unchanged(tmp_path, monkeypatch, mock_llm):
    """Restarting with the same documents must not throw cached answers away."""
    monkeypatch.chdir(tmp_path)
    documents_metadata.clear()
    thread_store.clear()

    with TestClient(app) as first:
        first.post("/chat", json={"query": "What is the capital of France?"})
    assert mock_llm.call_count == 1

    mock_llm.reset_mock()
    with TestClient(app) as second:
        response = second.post("/chat", json={"query": "What's France's capital city?"})

    assert response.json()["answer"] == "Mocked Gemini Response"
    mock_llm.assert_not_called()  # answered from the cache that survived the restart


def test_restart_keeps_cache_when_a_document_was_added(tmp_path, monkeypatch, mock_llm):
    """A document added while the app was off must not discard cached answers."""
    monkeypatch.chdir(tmp_path)
    documents_metadata.clear()
    thread_store.clear()

    with TestClient(app) as first:
        first.post("/chat", json={"query": "What is the capital of France?"})
    assert mock_llm.call_count == 1

    doc_dir = tmp_path / "documents" / "added-offline"
    doc_dir.mkdir(parents=True)
    (doc_dir / "original.txt").write_text("Ibuprofen dose is 200-400 mg.")

    mock_llm.reset_mock()
    with TestClient(app) as second:
        response = second.post("/chat", json={"query": "What's France's capital city?"})

    assert response.json()["answer"] == "Mocked Gemini Response"
    mock_llm.assert_not_called()  # still answered from the cache


def test_restart_forgets_only_answers_that_used_the_changed_document(tmp_path, monkeypatch, mock_llm):
    """Editing one document must leave the other documents' cached answers alone."""
    monkeypatch.chdir(tmp_path)
    documents_metadata.clear()
    thread_store.clear()

    with TestClient(app) as first:
        def upload(name, text):
            return first.post(
                "/upload/",
                files={"file": (name, io.BytesIO(text.encode()), "text/plain")},
                data={"chunk_size": 500, "chunk_overlap": 50},
            ).json()["document_id"]

        ibuprofen_id = upload(
            "ibuprofen.txt", "Ibuprofen adult dose is 200-400 mg every 4 to 6 hours."
        )
        upload("paracetamol.txt", "Paracetamol adult dose is 500-1000 mg every 4 to 6 hours.")
        first.post("/chat", json={"query": "What is the ibuprofen dose?", "k": 1})
        first.post("/chat", json={"query": "What is the paracetamol dose?", "k": 1})
    assert mock_llm.call_count == 2

    (tmp_path / "documents" / ibuprofen_id / "original.txt").write_text(
        "Ibuprofen adult dose is now 800 mg every 8 hours."
    )

    mock_llm.reset_mock()
    mock_llm.return_value = "Updated ibuprofen answer"
    with TestClient(app) as second:
        ibuprofen = second.post("/chat", json={"query": "What is the ibuprofen dose?", "k": 1})
        paracetamol = second.post("/chat", json={"query": "What is the paracetamol dose?", "k": 1})

    assert ibuprofen.json()["answer"] == "Updated ibuprofen answer"  # stale, forgotten
    assert paracetamol.json()["answer"] == "Mocked Gemini Response"  # still cached
    assert mock_llm.call_count == 1
