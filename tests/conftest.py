# root/tests/conftest.py
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from app import app  # Now works automatically because of pythonpath

@pytest.fixture
def client():
    # Mocking heavy components to stay within 'uv' speed
    app.state.encoder = MagicMock()
    app.state.encoder.encode.return_value = [[0.1] * 384] 
    app.state.collection = MagicMock()
    
    with TestClient(app) as c:
        yield c

@pytest.fixture
def mock_llm():
    with patch("app.ask_gemini") as mocked_ask:
        mocked_ask.return_value = "Mocked Gemini Response"
        yield mocked_ask