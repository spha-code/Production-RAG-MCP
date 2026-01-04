# tests/test_app.py
import pytest

def test_health_check(client):
    """
    Test the /health endpoint. 
    Expects 'healthy' because we fixed the GEMINI_CLIENT import in app.py.
    """
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "ProductionRAG-MCP"
    assert "llm_type" in data

def test_test_endpoint(client):
    """Test the basic heart-beat endpoint"""
    response = client.get("/test")
    assert response.status_code == 200
    assert response.json()["message"] == "Backend is running"
    assert "timestamp" in response.json()

def test_mcp_discovery(client):
    """Test the MCP tool definition endpoint for external LLM discovery"""
    response = client.get("/mcp")
    assert response.status_code == 200
    data = response.json()
    assert "tools" in data
    assert data["tools"][0]["name"] == "semantic_search"

def test_chat_endpoint(client, mock_llm):
    """
    Test the full RAG chat flow.
    The mock_llm fixture ensures we don't call the real Gemini API.
    """
    payload = {
        "query": "What is the dose of Paracetamol?",
        "thread_id": "test_thread_123",
        "k": 3
    }
    response = client.post("/chat", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    # This matches the 'Mocked Gemini Response' string in conftest.py
    assert "Mocked Gemini Response" in data["answer"]

def test_document_info_empty(client):
    """Test getting document info (should be empty but valid in a clean test env)"""
    response = client.get("/chat/documents/info")
    assert response.status_code == 200
    data = response.json()
    assert "total_documents" in data
    assert isinstance(data["documents"], list)

def test_chat_error_handling(client):
    """Test that the chat endpoint handles internal errors gracefully"""
    # Sending an empty/invalid payload to trigger a validation error
    response = client.post("/chat", json={})
    assert response.status_code == 422 # FastAPI validation error