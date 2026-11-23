# tests/test_app.py


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


def test_document_info_empty(client):
    response = client.get("/chat/documents/info")
    assert response.status_code == 200
    assert response.json() == {"total_documents": 0, "documents": []}


def test_chat_validation_error(client):
    response = client.post("/chat", json={})
    assert response.status_code == 422
