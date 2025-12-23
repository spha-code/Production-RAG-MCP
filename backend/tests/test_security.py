import pytest
from fastapi.testclient import TestClient

class TestSecurity:
    def test_sql_injection_prevention(self, test_client):
        """Test SQL injection prevention."""
        malicious_query = "test'; DROP TABLE documents; --"
        
        response = test_client.post("/chat", json={
            "query": malicious_query,
            "k": 1
        })
        
        # Should not crash or expose database errors
        assert response.status_code in [200, 400]
    
    def test_xss_prevention(self, test_client):
        """Test XSS prevention."""
        xss_query = "<script>alert('XSS')</script>"
        
        response = test_client.post("/chat", json={
            "query": xss_query,
            "k": 1
        })
        
        assert response.status_code == 200
        # Response should be properly escaped
        data = response.json()
        assert "<script>" not in str(data)
