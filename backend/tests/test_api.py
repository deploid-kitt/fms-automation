"""Tests for API endpoints."""
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


class TestHealthEndpoint:
    """Test health check endpoint."""
    
    def test_health_check_returns_200(self, client):
        """Test health endpoint returns 200."""
        response = client.get("/api/health")
        assert response.status_code == 200
    
    def test_health_check_returns_status(self, client):
        """Test health endpoint returns status."""
        response = client.get("/api/health")
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data


class TestRootEndpoint:
    """Test root endpoint."""
    
    def test_root_returns_app_info(self, client):
        """Test root endpoint returns app info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "FMS Automation"
        assert "version" in data


class TestUploadEndpoint:
    """Test video upload endpoint."""
    
    def test_upload_rejects_invalid_content_type(self, client):
        """Test that invalid file types are rejected."""
        response = client.post(
            "/api/upload",
            files={"file": ("test.txt", b"not a video", "text/plain")}
        )
        assert response.status_code == 400
        assert "Invalid file type" in response.json()["detail"]


class TestStatusEndpoint:
    """Test status endpoint."""
    
    def test_status_returns_404_for_unknown_job(self, client):
        """Test that unknown job ID returns 404."""
        response = client.get("/api/status/nonexistent-job-id")
        assert response.status_code == 404


class TestReportEndpoint:
    """Test report endpoint."""
    
    def test_report_returns_404_for_unknown_job(self, client):
        """Test that unknown job ID returns 404."""
        response = client.get("/api/report/nonexistent-job-id")
        assert response.status_code == 404
