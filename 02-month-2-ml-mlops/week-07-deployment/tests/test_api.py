"""
Test Suite for Churn Prediction API

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
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.0,
    "TotalCharges": 840.0
}

BATCH_PREDICTION_REQUEST = [
    {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.0,
        "TotalCharges": 840.0
    },
    {
        "gender": "Male",
        "SeniorCitizen": 1,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 48,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 90.0,
        "TotalCharges": 4320.0
    }
]

INVALID_PREDICTION_REQUEST = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": -5,  # Invalid: negative value
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.0,
    "TotalCharges": 840.0
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
    
    # Check required fields
    assert "status" in data
    assert "model_loaded" in data
    assert "model_version" in data
    assert "timestamp" in data
    
    # Verify status values
    assert data["status"] in ["healthy", "unhealthy"]
    assert isinstance(data["model_loaded"], bool)
    assert data["model_version"] == "1.0.0"


def test_health_check_response_structure(test_client):
    """Test that health check returns valid response structure."""
    response = test_client.get("/health")
    data = response.json()
    
    # Verify timestamp format (should be ISO format)
    try:
        datetime.fromisoformat(data["timestamp"])
    except ValueError:
        pytest.fail("Timestamp is not in valid ISO format")


# ==================== Root Endpoint Tests ====================

def test_root_endpoint(test_client):
    """
    Test the root endpoint.
    
    Verifies that:
    - Endpoint returns 200 status code
    - Response contains API information
    """
    response = test_client.get("/")
    
    assert response.status_code == 200
    data = response.json()
    
    # Check required fields
    assert "message" in data
    assert "documentation" in data
    assert "health_check" in data
    
    # Verify values
    assert data["message"] == "Churn Prediction API"
    assert data["documentation"] == "/docs"
    assert data["health_check"] == "/health"


# ==================== Prediction Endpoint Tests ====================

def test_predict_valid_request(test_client):
    """
    Test prediction with valid request.
    
    Verifies that:
    - Valid request returns 200 status
    - Response contains prediction and probability
    - Values are in expected ranges
    """
    response = test_client.post("/predict", json=VALID_PREDICTION_REQUEST)
    
    assert response.status_code == 200
    data = response.json()
    
    # Check response structure
    assert "churn_prediction" in data
    assert "churn_probability" in data
    assert "model_version" in data
    assert "timestamp" in data
    
    # Validate prediction values
    assert data["churn_prediction"] in [0, 1]
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["model_version"] == "1.0.0"


def test_predict_high_churn_risk(test_client):
    """
    Test prediction for high churn risk customer.
    
    Customer with:
    - Short tenure
    - Month-to-month contract
    - No security services
    - High monthly charges
    """
    high_risk_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.0,
        "TotalCharges": 190.0
    }
    
    response = test_client.post("/predict", json=high_risk_customer)
    
    assert response.status_code == 200
    data = response.json()
    
    # High risk customer should have high churn probability
    assert data["churn_probability"] > 0.5
    assert data["churn_prediction"] == 1


def test_predict_low_churn_risk(test_client):
    """
    Test prediction for low churn risk customer.
    
    Customer with:
    - Long tenure
    - Two year contract
    - Security services
    - Automatic payment
    """
    low_risk_customer = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 60,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 65.0,
        "TotalCharges": 3900.0
    }
    
    response = test_client.post("/predict", json=low_risk_customer)
    
    assert response.status_code == 200
    data = response.json()
    
    # Low risk customer should have low churn probability
    assert data["churn_probability"] < 0.5
    assert data["churn_prediction"] == 0


def test_predict_missing_field(test_client):
    """
    Test prediction with missing optional field.
    
    Should use default values and return prediction.
    """
    # Missing some optional fields
    incomplete_request = {
        "tenure": 12,
        "MonthlyCharges": 70.0,
        "TotalCharges": 840.0,
        "Contract": "Month-to-month"
    }
    
    response = test_client.post("/predict", json=incomplete_request)
    
    # Should use defaults for missing fields
    assert response.status_code == 200
    data = response.json()
    assert "churn_prediction" in data


def test_predict_invalid_contract(test_client):
    """
    Test prediction with invalid contract type.
    
    Note: The model may accept any string for Contract field.
    In production, consider adding validation for enum values.
    """
    invalid_request = VALID_PREDICTION_REQUEST.copy()
    invalid_request["Contract"] = "InvalidContract"
    
    response = test_client.post("/predict", json=invalid_request)
    
    # API accepts any string value (model handles encoding)
    # In production, add enum validation to return 422
    assert response.status_code == 200


def test_predict_negative_tenure(test_client):
    """
    Test prediction with negative tenure.
    """
    invalid_request = VALID_PREDICTION_REQUEST.copy()
    invalid_request["tenure"] = -5
    
    response = test_client.post("/predict", json=invalid_request)
    
    # The API accepts this but model might handle it
    assert response.status_code in [200, 422]


# ==================== Batch Prediction Tests ====================

def test_batch_predict_valid(test_client):
    """
    Test batch prediction with valid requests.
    
    Verifies that:
    - Valid batch returns 200 status
    - Response contains all predictions
    - Each prediction has required fields
    """
    response = test_client.post("/predict/batch", json={"customers": BATCH_PREDICTION_REQUEST})
    
    assert response.status_code == 200
    data = response.json()
    
    # Check response structure
    assert "predictions" in data
    assert len(data["predictions"]) == 2
    
    # Check each prediction
    for pred in data["predictions"]:
        assert "churn_prediction" in pred
        assert "churn_probability" in pred
        assert 0.0 <= pred["churn_probability"] <= 1.0


def test_batch_predict_empty(test_client):
    """
    Test batch prediction with empty list.
    """
    response = test_client.post("/predict/batch", json={"customers": []})
    
    # Empty list should return empty predictions
    assert response.status_code == 200
    data = response.json()
    assert data["predictions"] == []


def test_batch_predict_single(test_client):
    """
    Test batch prediction with single item.
    """
    response = test_client.post("/predict/batch", json={"customers": [VALID_PREDICTION_REQUEST]})
    
    assert response.status_code == 200
    data = response.json()
    
    assert len(data["predictions"]) == 1
    assert "churn_prediction" in data["predictions"][0]


# ==================== Error Handling Tests ====================

def test_invalid_endpoint(test_client):
    """
    Test request to non-existent endpoint.
    
    Verifies that API returns 404 for unknown endpoints.
    """
    response = test_client.get("/invalid_endpoint")
    assert response.status_code == 404


def test_wrong_method(test_client):
    """
    Test using wrong HTTP method.
    
    Verifies that API returns 405 for unsupported methods.
    """
    response = test_client.get("/predict")  # GET instead of POST
    assert response.status_code == 405


def test_malformed_json(test_client):
    """
    Test with malformed JSON body.
    
    Verifies that API returns 422 for invalid JSON.
    """
    response = test_client.post(
        "/predict",
        data="not valid json",
        headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422


# ==================== Integration Tests ====================

def test_full_workflow(test_client):
    """
    Integration test covering the full workflow:
    1. Check health
    2. Make single prediction
    3. Make batch prediction
    """
    # Step 1: Health check
    health_response = test_client.get("/health")
    assert health_response.status_code == 200
    assert health_response.json()["model_loaded"] is True
    
    # Step 2: Single prediction
    single_response = test_client.post("/predict", json=VALID_PREDICTION_REQUEST)
    assert single_response.status_code == 200
    single_data = single_response.json()
    assert "churn_prediction" in single_data
    
    # Step 3: Batch prediction
    batch_response = test_client.post("/predict/batch", json={"customers": BATCH_PREDICTION_REQUEST})
    assert batch_response.status_code == 200
    batch_data = batch_response.json()
    assert len(batch_data["predictions"]) == 2


# ==================== Performance Tests ====================

def test_response_time(test_client):
    """
    Test that endpoints respond within acceptable time limits.
    
    Note: These are basic sanity checks, not comprehensive load tests.
    """
    import time
    
    # Test health endpoint
    start_time = time.time()
    response = test_client.get("/health")
    elapsed_time = time.time() - start_time
    
    assert response.status_code == 200
    assert elapsed_time < 1.0  # Should respond within 1 second
    
    # Test prediction endpoint
    start_time = time.time()
    response = test_client.post("/predict", json=VALID_PREDICTION_REQUEST)
    elapsed_time = time.time() - start_time
    
    assert response.status_code == 200
    assert elapsed_time < 2.0  # Prediction should complete within 2 seconds


# ==================== Main ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
