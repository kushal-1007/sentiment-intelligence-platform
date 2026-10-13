import pytest
import sys
from pathlib import Path
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.api.main import app


# ── Session-scoped client — loads model ONCE for all tests ──
@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:   # 'with' triggers lifespan startup
        yield c                  # model loads here, stays loaded for all tests


# ════════════════════════════════════════
# Test 1: Health check
# ════════════════════════════════════════
def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


# ════════════════════════════════════════
# Test 2: Positive review → high star
# ════════════════════════════════════════
def test_predict_positive_review(client):
    response = client.post("/predict", json={
        "text": "Absolutely incredible food, amazing staff, best restaurant in the city!"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_class"] in [3, 4]
    assert data["confidence"] > 0.5
    assert "all_probabilities" in data
    assert len(data["all_probabilities"]) == 5


# ════════════════════════════════════════
# Test 3: Negative review → low star
# ════════════════════════════════════════
def test_predict_negative_review(client):
    response = client.post("/predict", json={
        "text": "Absolutely terrible food, worst service ever, never coming back!"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_class"] in [0, 1]
    assert data["confidence"] > 0.5


# ════════════════════════════════════════
# Test 4: Too short text → 422
# ════════════════════════════════════════
def test_predict_too_short_text():
    with TestClient(app) as c:
        response = c.post("/predict", json={"text": "ok"})
        assert response.status_code == 422


# ════════════════════════════════════════
# Test 5: Empty text → 422
# ════════════════════════════════════════
def test_predict_empty_text():
    with TestClient(app) as c:
        response = c.post("/predict", json={"text": ""})
        assert response.status_code == 422


# ════════════════════════════════════════
# Test 6: Batch prediction
# ════════════════════════════════════════
def test_predict_batch(client):
    texts = [
        "Amazing food and great service!",
        "Terrible experience, never returning.",
        "It was okay, nothing special."
    ]
    response = client.post("/predict/batch", json={"texts": texts})
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 3
    assert len(data["results"]) == 3
    for result in data["results"]:
        assert "predicted_class" in result
        assert "confidence" in result
        assert 0 <= result["predicted_class"] <= 4
        assert 0.0 <= result["confidence"] <= 1.0


# ════════════════════════════════════════
# Test 7: Feedback endpoint
# ════════════════════════════════════════
def test_feedback_endpoint(client):
    response = client.post("/feedback", json={
        "text": "The food was okay but service was slow",
        "predicted_class": 2,
        "correct_class": 1,
        "confidence": 0.55
    })
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "corrections_total" in data
    assert data["corrections_total"] >= 1


# ════════════════════════════════════════
# Test 8: Feedback stats
# ════════════════════════════════════════
def test_feedback_stats_endpoint(client):
    response = client.get("/feedback/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_corrections" in data
    assert "pending_retraining" in data
    assert "retraining_threshold" in data
    assert "ready_to_retrain" in data
    assert data["retraining_threshold"] == 20


# ════════════════════════════════════════
# Test 9: Model info
# ════════════════════════════════════════
def test_model_info_endpoint(client):
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "distilbert-base-uncased"
    assert data["num_labels"] == 5
    assert len(data["label_names"]) == 5
    assert data["evaluation_metrics"] is not None
    assert "distilbert" in data["evaluation_metrics"]
    metrics = data["evaluation_metrics"]["distilbert"]
    assert metrics["macro_f1"] > 0.5
    assert metrics["roc_auc_macro"] > 0.8


# ════════════════════════════════════════
# Test 10: Batch empty list → 422
# ════════════════════════════════════════
def test_predict_batch_empty_list(client):
    response = client.post("/predict/batch", json={"texts": []})
    assert response.status_code == 422