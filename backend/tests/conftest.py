import pytest
import tempfile
import shutil
from pathlib import Path
from fastapi.testclient import TestClient
from unittest.mock import Mock
import chromadb
from sentence_transformers import SentenceTransformer

from backend.app import app

@pytest.fixture
def test_client():
    """Create a test client for FastAPI app."""
    return TestClient(app)

@pytest.fixture
def temp_chroma_db():
    """Create a temporary ChromaDB for testing."""
    temp_dir = tempfile.mkdtemp()
    client = chromadb.PersistentClient(path=temp_dir)
    
    yield client
    
    # Cleanup
    shutil.rmtree(temp_dir)

@pytest.fixture
def mock_sentence_transformer():
    """Mock sentence transformer for unit tests."""
    mock_encoder = Mock()
    # Return predictable embeddings for testing
    mock_encoder.encode.return_value = [[0.1, 0.2, 0.3, 0.4, 0.5]]
    return mock_encoder

@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        "GDPR applies to any company processing EU residents data.",
        "Paracetamol typical dose is 500-1000 mg.",
        "AWS free-tier includes 750 h of t2.micro per month."
    ]
