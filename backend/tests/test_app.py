import pytest
from unittest.mock import patch, Mock

class TestChatEndpoint:
    def test_chat_endpoint_success(self, test_client, sample_documents):
        """Test successful chat query."""
        response = test_client.post("/chat", json={
            "query": "What is GDPR?",
            "k": 2
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "chunks" in data
        assert isinstance(data["chunks"], list)
    
    def test_chat_endpoint_invalid_input(self, test_client):
        """Test chat endpoint with invalid input."""
        response = test_client.post("/chat", json={
            "query": "",  # Empty query
            "k": 2
        })
        
        assert response.status_code == 422
    
    def test_chat_endpoint_embedding_failure(self, test_client):
        """Test chat endpoint when embedding fails."""
        with patch('sentence_transformers.SentenceTransformer.encode') as mock_encode:
            mock_encode.side_effect = Exception("Embedding failed")
            
            response = test_client.post("/chat", json={
                "query": "test query",
                "k": 2
            })
            
            assert response.status_code == 500

class TestMCPtoolsEndpoint:
    def test_mcp_tools_endpoint(self, test_client):
        """Test MCP tools endpoint returns correct structure."""
        response = test_client.get("/mcp")
        
        assert response.status_code == 200
        data = response.json()
        assert "tools" in data
        assert isinstance(data["tools"], list)
        
        # Check semantic_search tool structure
        semantic_tool = next((tool for tool in data["tools"] 
                            if tool["name"] == "semantic_search"), None)
        assert semantic_tool is not None
        assert "inputSchema" in semantic_tool

class TestHealthEndpoints:
    def test_app_startup(self, test_client):
        """Test that the app starts correctly."""
        response = test_client.get("/docs")
        assert response.status_code == 200
    
    def test_openapi_schema(self, test_client):
        """Test OpenAPI schema generation."""
        response = test_client.get("/openapi.json")
        assert response.status_code == 200
