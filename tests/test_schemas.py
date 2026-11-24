# tests/test_schemas.py - the request/response models /chat actually uses
import pytest
from pydantic import ValidationError

from backend.app import ChatRequest, ChatResponse


def test_chat_request_defaults():
    request = ChatRequest(query="What is Paracetamol?")
    assert request.query == "What is Paracetamol?"
    assert request.k == 3
    assert request.thread_id == "default"


def test_chat_request_custom_values():
    request = ChatRequest(query="What is GDPR?", k=5, thread_id="abc")
    assert request.k == 5
    assert request.thread_id == "abc"


def test_chat_request_invalid():
    with pytest.raises(ValidationError):
        ChatRequest(k=3)  # missing query
    with pytest.raises(ValidationError):
        ChatRequest(query="Test", k="not-an-integer")


def test_chat_response():
    assert ChatResponse(answer="Paris").answer == "Paris"
    with pytest.raises(ValidationError):
        ChatResponse()
