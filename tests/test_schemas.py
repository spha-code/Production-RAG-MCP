import pytest
from pydantic import ValidationError
from rag.schemas import ChatRequest, ChatResponse

def test_chat_request_valid():
    """Test ChatRequest with valid data and default k value"""
    data = {"query": "What is Paracetamol?"}
    request = ChatRequest(**data)
    
    assert request.query == "What is Paracetamol?"
    assert request.k == 3  # Check default value

def test_chat_request_custom_k():
    """Test ChatRequest with a custom k value"""
    data = {"query": "What is GDPR?", "k": 5}
    request = ChatRequest(**data)
    
    assert request.query == "What is GDPR?"
    assert request.k == 5

def test_chat_request_invalid_types():
    """Test ChatRequest with invalid types to trigger validation errors"""
    # Test missing required field 'query'
    with pytest.raises(ValidationError):
        ChatRequest(k=3)
    
    # Test invalid type for 'k' (should be int)
    with pytest.raises(ValidationError):
        ChatRequest(query="Test", k="not-an-integer")

def test_chat_response_valid():
    """Test ChatResponse with valid list of strings"""
    chunks = ["Chunk 1 content", "Chunk 2 content"]
    response = ChatResponse(chunks=chunks)
    
    assert len(response.chunks) == 2
    assert response.chunks[0] == "Chunk 1 content"

def test_chat_response_empty():
    """Test ChatResponse with an empty list (edge case)"""
    response = ChatResponse(chunks=[])
    assert response.chunks == []

def test_chat_response_invalid():
    """Test ChatResponse with a non-list object"""
    with pytest.raises(ValidationError):
        # Passing a string instead of a list of strings
        ChatResponse(chunks="This is not a list")