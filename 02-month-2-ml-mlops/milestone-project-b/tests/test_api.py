"""
Comprehensive Test Suite for Churn Prediction API

This module contains unit and integration tests for the ML microservice.

Coverage:
- Health endpoint tests
- Prediction endpoint tests (single and batch)
- Error handling tests
- Input validation tests

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

VALID_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "Yes",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "One year",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Mailed check",
    "MonthlyCharges": 65.0,
    "TotalCharges": 780.0
}

VALID_PREDICTION_REQUEST = {
    "customer_id": "test_cust_001",
    "data": VALID_CUSTOMER
}

BATCH_REQUEST = {
    "customers": [
        {
            "customer_id": "cust_001",
            "data": VALID_CUSTOMER
        },
        {
            "customer_id": "cust_002",
            "data": {
                **VALID_CUSTOMER,
                "tenure": 1,
                "Contract": "Month-to-month",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 90.0
            }
        }
    ]
}

INVALID_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 5,  # Invalid: should be 0 or 1
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "Yes",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "One year",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Mailed check",
    "MonthlyCharges": 65.0,
    "TotalCharges": 780.0
}

MISSING_FIELD_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    # Missing Partner field
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "Yes",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "One year",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Mailed check",
    "MonthlyCharges": 65.0,
    "TotalCharges": 780.0
}

NEGATIVE_CHARGES_CUSTOMER = {
    **VALID_CUSTOMER,
    "MonthlyCharges": -50.0  # Invalid: negative value
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


# ==================== Root Endpoint Tests ====================

class TestRootEndpoint:
    """Tests for the root endpoint."""
    
    def test_root_returns_api_info(self, test_client):
        """Test that root endpoint returns API information."""
        response = test_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "documentation" in data
        assert data["documentation"] == "/docs"


# ==================== Health Endpoint Tests ====================

class TestHealthEndpoint:
    """Tests for the health check endpoint."""
    
    def test_health_check_returns_200(self, test_client):
        """Test that health check returns 200 status."""
        response = test_client.get("/health")
        
        assert response.status_code == 200
    
    def test_health_check_response_structure(self, test_client):
        """Test that health check returns expected structure."""
        response = test_client.get("/health")
        data = response.json()
        
        assert "status" in data
        assert "model_loaded" in data
        assert "model_version" in data
        assert "timestamp" in data
    
    def test_health_check_status_values(self, test_client):
        """Test that status is either healthy or unhealthy."""
        response = test_client.get("/health")
        data = response.json()
        
        assert data["status"] in ["healthy", "unhealthy"]
        assert isinstance(data["model_loaded"], bool)


# ==================== Model Info Endpoint Tests ====================

class TestModelInfoEndpoint:
    """Tests for the model info endpoint."""
    
    def test_model_info_returns_200(self, test_client):
        """Test that model info returns 200 status."""
        response = test_client.get("/model/info")
        
        assert response.status_code == 200
    
    def test_model_info_response_structure(self, test_client):
        """Test that model info returns expected structure."""
        response = test_client.get("/model/info")
        data = response.json()
        
        assert "model_loaded" in data
        assert "model_version" in data
        assert "expected_features" in data
        assert "prediction_threshold" in data
        assert isinstance(data["expected_features"], list)
        assert len(data["expected_features"]) > 0


# ==================== Single Prediction Tests ====================

class TestSinglePrediction:
    """Tests for single prediction endpoint."""
    
    def test_predict_valid_request(self, test_client):
        """Test prediction with valid request."""
        response = test_client.post("/predict", json=VALID_PREDICTION_REQUEST)
        
        assert response.status_code == 200
        
        data = response.json()
        assert "churn_probability" in data
        assert "churn_prediction" in data
        assert "confidence" in data
        assert "model_version" in data
        assert "prediction_time" in data
        
        # Validate data types
        assert isinstance(data["churn_probability"], float)
        assert 0 <= data["churn_probability"] <= 1
        assert isinstance(data["churn_prediction"], bool)
        assert data["confidence"] in ["high", "medium", "low"]
    
    def test_predict_returns_customer_id(self, test_client):
        """Test that prediction returns customer_id."""
        response = test_client.post("/predict", json=VALID_PREDICTION_REQUEST)
        
        data = response.json()
        assert data["customer_id"] == "test_cust_001"
    
    def test_predict_without_customer_id(self, test_client):
        """Test prediction without customer_id."""
        request = {"data": VALID_CUSTOMER}
        response = test_client.post("/predict", json=request)
        
        assert response.status_code == 200
        data = response.json()
        assert data["customer_id"] is None
    
    def test_predict_month_to_month_high_risk(self, test_client):
        """Test that month-to-month contracts have higher churn probability."""
        high_risk_customer = {
            "customer_id": "high_risk_001",
            "data": {
                **VALID_CUSTOMER,
                "Contract": "Month-to-month",
                "tenure": 1,
                "PaymentMethod": "Electronic check"
            }
        }
        
        response = test_client.post("/predict", json=high_risk_customer)
        data = response.json()
        
        assert response.status_code == 200
        # Month-to-month customers should generally have higher churn prob
        assert "churn_probability" in data


# ==================== Batch Prediction Tests ====================

class TestBatchPrediction:
    """Tests for batch prediction endpoint."""
    
    def test_batch_predict_valid(self, test_client):
        """Test batch prediction with valid request."""
        response = test_client.post("/predict/batch", json=BATCH_REQUEST)
        
        assert response.status_code == 200
        
        data = response.json()
        assert "predictions" in data
        assert "total" in data
        assert "successful" in data
        assert "failed" in data
        
        assert data["total"] == 2
        assert data["successful"] == 2
        assert data["failed"] == 0
        assert len(data["predictions"]) == 2
    
    def test_batch_predict_returns_all_predictions(self, test_client):
        """Test that batch prediction returns all predictions."""
        response = test_client.post("/predict/batch", json=BATCH_REQUEST)
        
        data = response.json()
        
        for pred in data["predictions"]:
            assert "churn_probability" in pred
            assert "churn_prediction" in pred
            assert "confidence" in pred
    
    def test_batch_predict_empty_list(self, test_client):
        """Test batch prediction with empty list."""
        response = test_client.post("/predict/batch", json={"customers": []})
        
        assert response.status_code == 422  # Validation error
    
    def test_batch_predict_single_item(self, test_client):
        """Test batch prediction with single item."""
        single_request = {"customers": [VALID_PREDICTION_REQUEST]}
        response = test_client.post("/predict/batch", json=single_request)
        
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert data["successful"] == 1


# ==================== Input Validation Tests ====================

class TestInputValidation:
    """Tests for input validation."""
    
    def test_predict_invalid_senior_citizen(self, test_client):
        """Test prediction with invalid SeniorCitizen value."""
        request = {
            "customer_id": "test_001",
            "data": INVALID_CUSTOMER
        }
        response = test_client.post("/predict", json=request)
        
        assert response.status_code == 422
    
    def test_predict_missing_field(self, test_client):
        """Test prediction with missing required field."""
        request = {
            "customer_id": "test_001",
            "data": MISSING_FIELD_CUSTOMER
        }
        response = test_client.post("/predict", json=request)
        
        assert response.status_code == 422
    
    def test_predict_negative_charges(self, test_client):
        """Test prediction with negative charges."""
        request = {
            "customer_id": "test_001",
            "data": NEGATIVE_CHARGES_CUSTOMER
        }
        response = test_client.post("/predict", json=request)
        
        assert response.status_code == 422
    
    def test_predict_invalid_tenure(self, test_client):
        """Test prediction with invalid tenure."""
        invalid_customer = {**VALID_CUSTOMER, "tenure": 200}
        request = {"data": invalid_customer}
        
        response = test_client.post("/predict", json=request)
        assert response.status_code == 422


# ==================== Documentation Tests ====================

class TestDocumentation:
    """Tests for API documentation."""
    
    def test_swagger_ui_accessible(self, test_client):
        """Test that Swagger UI is accessible."""
        response = test_client.get("/docs")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_openapi_schema_accessible(self, test_client):
        """Test that OpenAPI schema is accessible."""
        response = test_client.get("/openapi.json")
        
        assert response.status_code == 200
        
        data = response.json()
        assert "openapi" in data
        assert "paths" in data
        assert "/predict" in data["paths"]
        assert "/health" in data["paths"]
    
    def test_redoc_accessible(self, test_client):
        """Test that ReDoc is accessible."""
        response = test_client.get("/redoc")
        
        assert response.status_code == 200


# ==================== Error Handling Tests ====================

class TestErrorHandling:
    """Tests for error handling."""
    
    def test_invalid_endpoint_returns_404(self, test_client):
        """Test that invalid endpoint returns 404."""
        response = test_client.get("/invalid_endpoint")
        
        assert response.status_code == 404
    
    def test_invalid_method_returns_405(self, test_client):
        """Test that invalid HTTP method returns 405."""
        response = test_client.get("/predict")  # POST expected
        
        assert response.status_code == 405
    
    def test_malformed_json_returns_422(self, test_client):
        """Test that malformed JSON returns 422."""
        response = test_client.post(
            "/predict",
            data="not valid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422


# ==================== Integration Tests ====================

class TestIntegration:
    """Integration tests covering full workflow."""
    
    def test_full_workflow(self, test_client):
        """
        Test the complete workflow:
        1. Check health
        2. Get model info
        3. Make single prediction
        4. Make batch prediction
        """
        # Step 1: Health check
        health_response = test_client.get("/health")
        assert health_response.status_code == 200
        health_data = health_response.json()
        assert health_data["status"] == "healthy"
        
        # Step 2: Get model info
        info_response = test_client.get("/model/info")
        assert info_response.status_code == 200
        
        # Step 3: Single prediction
        single_response = test_client.post("/predict", json=VALID_PREDICTION_REQUEST)
        assert single_response.status_code == 200
        single_data = single_response.json()
        assert single_data["churn_probability"] is not None
        
        # Step 4: Batch prediction
        batch_response = test_client.post("/predict/batch", json=BATCH_REQUEST)
        assert batch_response.status_code == 200
        batch_data = batch_response.json()
        assert batch_data["successful"] == 2


# ==================== Performance Tests ====================

class TestPerformance:
    """Basic performance tests."""
    
    def test_prediction_response_time(self, test_client):
        """Test that prediction responds within acceptable time."""
        import time
        
        start_time = time.time()
        response = test_client.post("/predict", json=VALID_PREDICTION_REQUEST)
        elapsed_time = time.time() - start_time
        
        assert response.status_code == 200
        assert elapsed_time < 5.0  # Should complete within 5 seconds
    
    def test_health_check_fast(self, test_client):
        """Test that health check is fast."""
        import time
        
        start_time = time.time()
        response = test_client.get("/health")
        elapsed_time = time.time() - start_time
        
        assert response.status_code == 200
        assert elapsed_time < 1.0  # Should complete within 1 second


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
