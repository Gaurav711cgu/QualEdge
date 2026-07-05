import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"

def test_overview_stats():
    response = client.get("/api/overview/stats")
    assert response.status_code == 200
    data = response.json()
    assert "compressionRatio" in data
    assert "accuracyDelta" in data
    assert "driftStatus" in data

def test_compression_benchmarks():
    response = client.get("/api/compression/benchmarks")
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 3
    model_names = {b["modelName"] for b in data}
    assert "mobilenet_v2" in model_names
    assert "whisper_tiny" in model_names
    assert "phi_3_mini" in model_names

def test_aihub_jobs():
    response = client.get("/api/aihub/jobs")
    assert response.status_code == 200
    # Returns list of jobs, initially empty or populated after compression
    assert isinstance(response.json(), list)

def test_router_endpoint():
    payload = {
        "query": "What is the capital of France?",
        "pathway": "tfidf",
        "forceDegrade": False
    }
    response = client.post("/api/router/route", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["decision"] == "on_device"
    assert "text" in data
    assert data["estimatedCloudCostUsd"] == 0.0

def test_router_endpoint_retry():
    # Force degradation
    payload = {
        "query": "Draft a polite email asking for feedback on my project.", # Moderate reasoning
        "pathway": "tfidf",
        "forceDegrade": True
    }
    response = client.post("/api/router/route", json=payload)
    assert response.status_code == 200
    data = response.json()
    # It fails self-verification on local output, and since it is moderate reasoning, escalates to cloud retry
    assert data["decision"] == "cloud"
    assert "Cloud Model" in data["device"]
    assert data["estimatedCloudCostUsd"] > 0.0

def test_router_sweep():
    response = client.get("/api/router/sweep")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 10
    assert "falseNegativeRate" in data[0]
