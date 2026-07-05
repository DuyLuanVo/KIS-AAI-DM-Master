"""
Tests for video search endpoints
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


class TestVideoSearch:
    """Test cases for video search endpoints"""
    
    def test_search_videos_post(self):
        """Test POST /api/v1/videos/search"""
        request_data = {
            "query_text": ["a person walking"]
        }
        
        response = client.post("/api/v1/videos/search", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "results" in data
        
        # Check results structure
        if data["results"]:
            result = data["results"][0]
            assert "original_id" in result
            assert "jpg_path" in result
            assert "frame_idx" in result
    
    def test_search_videos_response_structure(self):
        """Test that response has correct structure"""
        request_data = {
            "query_text": ["car driving"]
        }
        
        response = client.post("/api/v1/videos/search", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data["results"], list)
        
        # Check each result has required fields
        for result in data["results"]:
            assert isinstance(result["original_id"], str)
            assert isinstance(result["jpg_path"], str)
            assert isinstance(result["frame_idx"], int)
    
    def test_search_videos_returns_data(self):
        """Test that search returns some data"""
        request_data = {
            "query_text": ["dog running"]
        }
        
        response = client.post("/api/v1/videos/search", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["results"]) > 0
    
    def test_search_videos_empty_query(self):
        """Test search with empty query text"""
        request_data = {
            "query_text": []
        }
        
        response = client.post("/api/v1/videos/search", json=request_data)
        assert response.status_code == 200  # Should still work with empty query
    
    def test_search_videos_long_query(self):
        """Test search with long query text"""
        request_data = {
            "query_text": ["a person walking in the park with a dog on a sunny day"]
        }
        
        response = client.post("/api/v1/videos/search", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "results" in data
