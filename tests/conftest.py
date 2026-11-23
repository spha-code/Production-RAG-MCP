# tests/conftest.py
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from backend.app import app, thread_store
from backend.routes.upload import documents_metadata


@pytest.fixture
def client(tmp_path, monkeypatch):
    """
    Run the real app inside an empty temp folder.

    The app uses relative paths (./chroma, ./documents), so changing the
    working directory keeps tests away from your real database and uploads.
    The real embedding model is used (it's local and cached after first run).
    """
    monkeypatch.chdir(tmp_path)
    documents_metadata.clear()
    thread_store.clear()

    with TestClient(app) as c:
        yield c

    documents_metadata.clear()


@pytest.fixture
def mock_llm():
    """Replace the Gemini call so tests never hit the network or use quota."""
    with patch("backend.app.ask_llm") as mocked_ask:
        mocked_ask.return_value = "Mocked Gemini Response"
        yield mocked_ask
