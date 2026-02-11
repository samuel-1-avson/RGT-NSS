"""
Test Suite for House Price Prediction API

This module contains unit and integration tests for the FastAPI application.

Run tests:
    pytest tests/test_api.py -v

Run with coverage:
    pytest tests/test_api.py --cov=api --cov-report=html
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from api.main import app, load_model

# Create test client
client = TestClient(app)


# ==================== Test Data ====================

VALID_PREDICTION_REQUEST = {
    "LotArea": 8450,
    "OverallQual": 7,
    "OverallCond": 5,
    "YearBuilt": 2003,
    "TotalBsmtSF": 856,
    "GrLivArea": 1710,
    "GarageCars": 2,
    "GarageArea": 548
}

BATCH_PREDICTION_REQUEST = [
    {
        "LotArea": 8450,
        "OverallQual": 7,
        "OverallCond": 5,
        "YearBuilt": 2003,
        "TotalBsmtSF": 856,
        "GrLivArea": 1710,
        "GarageCars": 2,
        "GarageArea": 548
    },
    {
        "LotArea": 9600,
        "OverallQual": 6,
        "OverallCond": 8,
        "YearBuilt": 1976,
        "TotalBsmtSF": 1262,
        "GrLivArea": 1262,
        "GarageCars": 2,
        "GarageArea": 460
    }
]

INVALID_PREDICTION_REQUEST = {
    "LotArea": -100,  # Invalid: negative value
    "OverallQual": 7,
    "OverallCond": 5,
    "YearBuilt": 2003,
    "TotalBsmtSF": 856,
    "GrLivArea": 1710,
    "GarageCars": 2,
    "GarageArea": 548
}


# ==================== Fixtures ====================

@pytest.fixture(scope="module")
def test_client():
    """Create a test client for the API."""
    return TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def setup_model():
    """Ensure model is loaded before running tests."""
    load_model()


# ==================== Health Endpoint Tests ====================

def test_health_check(test_client):
    """
    Test the health check endpoint.
    
    Verifies that:
    - Endpoint returns 200 status code
    - Response contains expected fields
    - Model status is correctly reported
    """
    response = test_client.get("/health")
    
    assert response.status_code == 200
    
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "model_version" in data
    assert "timestamp" in data
    
    # Status should be healthy or unhealthy
    assert data["status"] in ["healthy", "unhealthy"]
    assert isinstance(data["model_loaded"], bool)


def test_root_endpoint(test_client):
    """
    Test the root endpoint.
    
    Verifies that the root endpoint returns API information.
    """
    response = test_client.get("/")
    
    assert response.status_code == 200
    
    data = response.json()
    assert "name" in data
    assert "version" in data
    assert "documentation" in data


# ==================== Prediction Endpoint Tests ====================

def test_predict_valid_request(test_client):
    """
    Test prediction with valid request data.
    
    Verifies that:
    - Valid request returns 200 status code
    - Response contains predicted_price
    - Response contains prediction_interval
    - Response contains model_version and timestamp
    """
    response = test_client.post("/predict", json=VALID_PREDICTION_REQUEST)
    
    assert response.status_code == 200
    
    data = response.json()
    assert "predicted_price" in data
    assert "prediction_interval" in data
    assert "model_version" in data
    assert "timestamp" in data
    
    # Validate prediction value
    assert isinstance(data["predicted_price"], float)
    assert data["predicted_price"] > 0
    
    # Validate prediction interval
    assert len(data["prediction_interval"]) == 2
    assert data["prediction_interval"][0] <= data["predicted_price"]
    assert data["predicted_price"] <= data["prediction_interval"][1]


def test_predict_invalid_lot_area(test_client):
    """
    Test prediction with invalid LotArea (negative value).
    
    Verifies that the API returns a validation error for invalid input.
    """
    response = test_client.post("/predict", json=INVALID_PREDICTION_REQUEST)
    
    assert response.status_code == 422  # Validation error
    
    data = response.json()
    assert "detail" in data


def test_predict_missing_field(test_client):
    """
    Test prediction with missing required field.
    
    Verifies that the API returns a validation error when required fields are missing.
    """
    incomplete_request = VALID_PREDICTION_REQUEST.copy()
    del incomplete_request["LotArea"]  # Remove required field
    
    response = test_client.post("/predict", json=incomplete_request)
    
    assert response.status_code == 422
    
    data = response.json()
    assert "detail" in data


def test_predict_invalid_quality(test_client):
    """
    Test prediction with invalid quality rating (outside 1-10 range).
    
    Verifies that the API validates the OverallQual range.
    """
    invalid_request = VALID_PREDICTION_REQUEST.copy()
    invalid_request["OverallQual"] = 15  # Outside valid range
    
    response = test_client.post("/predict", json=invalid_request)
    
    assert response.status_code == 422


def test_predict_invalid_year(test_client):
    """
    Test prediction with invalid year (in the future).
    
    Verifies that the API validates the YearBuilt range.
    """
    invalid_request = VALID_PREDICTION_REQUEST.copy()
    invalid_request["YearBuilt"] = 2030  # Future year
    
    response = test_client.post("/predict", json=invalid_request)
    
    assert response.status_code == 422


def test_predict_zero_lot_area(test_client):
    """
    Test prediction with zero LotArea (should fail gt=0 validation).
    """
    invalid_request = VALID_PREDICTION_REQUEST.copy()
    invalid_request["LotArea"] = 0
    
    response = test_client.post("/predict", json=invalid_request)
    
    assert response.status_code == 422


# ==================== Batch Prediction Tests ====================

def test_batch_predict_valid(test_client):
    """
    Test batch prediction with valid requests.
    
    Verifies that:
    - Batch request returns 200 status code
    - Response contains predictions array
    - Correct number of predictions returned
    """
    response = test_client.post("/predict/batch", json=BATCH_PREDICTION_REQUEST)
    
    assert response.status_code == 200
    
    data = response.json()
    assert "predictions" in data
    assert "count" in data
    assert data["count"] == len(BATCH_PREDICTION_REQUEST)
    assert len(data["predictions"]) == len(BATCH_PREDICTION_REQUEST)
    
    # Validate each prediction
    for pred in data["predictions"]:
        assert "predicted_price" in pred
        assert "prediction_interval" in pred
        assert isinstance(pred["predicted_price"], float)


def test_batch_predict_empty(test_client):
    """
    Test batch prediction with empty list.
    """
    response = test_client.post("/predict/batch", json=[])
    
    assert response.status_code == 200
    
    data = response.json()
    assert data["count"] == 0
    assert data["predictions"] == []


def test_batch_predict_single(test_client):
    """
    Test batch prediction with single item.
    """
    response = test_client.post("/predict/batch", json=[VALID_PREDICTION_REQUEST])
    
    assert response.status_code == 200
    
    data = response.json()
    assert data["count"] == 1
    assert len(data["predictions"]) == 1


# ==================== Model Info Tests ====================

def test_model_info(test_client):
    """
    Test the model info endpoint.
    
    Verifies that:
    - Endpoint returns 200 status code
    - Response contains model metadata
    """
    response = test_client.get("/model/info")
    
    assert response.status_code == 200
    
    data = response.json()
    assert "model_loaded" in data
    assert "version" in data
    assert "expected_features" in data
    assert isinstance(data["expected_features"], list)


# ==================== Documentation Tests ====================

def test_docs_endpoint(test_client):
    """
    Test that Swagger UI documentation is accessible.
    """
    response = test_client.get("/docs")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_openapi_schema(test_client):
    """
    Test that OpenAPI schema is accessible.
    """
    response = test_client.get("/openapi.json")
    assert response.status_code == 200
    
    data = response.json()
    assert "openapi" in data
    assert "paths" in data


# ==================== Edge Cases ====================

def test_predict_very_large_values(test_client):
    """
    Test prediction with very large values.
    
    Verifies that the API handles large but valid input values.
    """
    large_request = VALID_PREDICTION_REQUEST.copy()
    large_request["LotArea"] = 100000  # Very large lot
    large_request["GrLivArea"] = 10000  # Very large house
    
    response = test_client.post("/predict", json=large_request)
    
    # Should accept large values (though prediction might be unrealistic)
    assert response.status_code == 200


def test_predict_very_old_house(test_client):
    """
    Test prediction with very old house (edge of valid range).
    """
    old_request = VALID_PREDICTION_REQUEST.copy()
    old_request["YearBuilt"] = 1800  # Edge of valid range
    
    response = test_client.post("/predict", json=old_request)
    
    assert response.status_code == 200


# ==================== Integration Test ====================

def test_full_workflow(test_client):
    """
    Integration test covering the full workflow:
    1. Check health
    2. Get model info
    3. Make single prediction
    4. Make batch prediction
    """
    # Step 1: Health check
    health_response = test_client.get("/health")
    assert health_response.status_code == 200
    
    # Step 2: Get model info
    info_response = test_client.get("/model/info")
    assert info_response.status_code == 200
    
    # Step 3: Single prediction
    single_response = test_client.post("/predict", json=VALID_PREDICTION_REQUEST)
    assert single_response.status_code == 200
    
    # Step 4: Batch prediction
    batch_response = test_client.post("/predict/batch", json=BATCH_PREDICTION_REQUEST)
    assert batch_response.status_code == 200


# ==================== Performance Tests ====================

@pytest.mark.parametrize("endpoint,method", [
    ("/health", "GET"),
    ("/predict", "POST"),
    ("/model/info", "GET"),
])
def test_response_time(test_client, endpoint, method):
    """
    Test that endpoints respond within acceptable time limits.
    
    Note: These are basic sanity checks, not comprehensive load tests.
    """
    import time
    
    start_time = time.time()
    
    if method == "GET":
        response = test_client.get(endpoint)
    else:
        response = test_client.post(endpoint, json=VALID_PREDICTION_REQUEST)
    
    elapsed_time = time.time() - start_time
    
    assert response.status_code == 200
    assert elapsed_time < 5.0  # Should respond within 5 seconds


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
