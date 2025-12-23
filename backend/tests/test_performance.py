import pytest
import time
from concurrent.futures import ThreadPoolExecutor

class TestPerformance:
    def test_response_time(self, test_client):
        """Test API response time."""
        start_time = time.time()
        
        response = test_client.post("/chat", json={
            "query": "What is GDPR?",
            "k": 3
        })
        
        end_time = time.time()
        response_time = end_time - start_time
        
        assert response.status_code == 200
        assert response_time < 2.0  # Should respond within 2 seconds
    
    def test_memory_usage(self, test_client):
        """Test memory usage during operations."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Make multiple requests
        for _ in range(10):
            test_client.post("/chat", json={
                "query": "Test query",
                "k": 5
            })
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 50MB)
        assert memory_increase < 50 * 1024 * 1024
