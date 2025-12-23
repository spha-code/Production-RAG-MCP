import pytest
import tempfile
import shutil
from pathlib import Path
from fastapi.testclient import TestClient

class TestRAGIntegration:
    def test_full_rag_pipeline(self, test_client, temp_chroma_db):
        """Test complete RAG pipeline from query to response."""
        # Setup test data in ChromaDB
        collection = temp_chroma_db.create_collection("test_integration")
        
        # Add test documents
        test_docs = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Natural language processing enables computers to understand text."
        ]
        
        collection.add(
            documents=test_docs,
            ids=[f"doc_{i}" for i in range(len(test_docs))],
            embeddings=[[0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i, 0.5 * i] for i in range(len(test_docs))]
        )
        
        # Test query
        with patch('backend.app.chromadb.PersistentClient', return_value=temp_chroma_db):
            response = test_client.post("/chat", json={
                "query": "What is machine learning?",
                "k": 2
            })
            
            assert response.status_code == 200
            data = response.json()
            assert len(data["chunks"]) == 2

    def test_concurrent_queries(self, test_client):
        """Test handling of concurrent queries."""
        import concurrent.futures
        
        queries = [
            {"query": "What is AI?", "k": 1},
            {"query": "What is ML?", "k": 1},
            {"query": "What is NLP?", "k": 1}
        ]
        
        def make_request(query_data):
            return test_client.post("/chat", json=query_data)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(make_request, query) for query in queries]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        assert all(result.status_code == 200 for result in results)
