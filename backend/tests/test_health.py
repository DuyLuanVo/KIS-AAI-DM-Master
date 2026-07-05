"""
Tests for health endpoints
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert "status" in data
    assert "timestamp" in data
    assert "version" in data


def test_qdrant_health_check():
    """Test Qdrant health check endpoint"""
    response = client.get("/health/qdrant")
    # Note: This test might fail if Qdrant is not running
    # In a real test environment, you'd mock the Qdrant client
    assert response.status_code in [200, 503]


def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    
    data = response.json()
    assert "message" in data
    assert "version" in data
    assert "docs" in data
    assert "health" in data
