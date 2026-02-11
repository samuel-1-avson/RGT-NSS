"""
Week 7: Model Deployment with FastAPI
======================================

This module implements a REST API for serving the churn prediction model.
It provides endpoints for health checks and making predictions.

API Endpoints:
    GET  /health          - Health check endpoint
    POST /predict         - Single customer prediction
    POST /predict/batch   - Batch prediction for multiple customers

Features:
    - Input validation using Pydantic models
    - Automatic API documentation (Swagger UI at /docs)
    - Error handling with meaningful messages
    - Request/response logging
    - Model versioning

Model: Tuned Random Forest from Week 6
Framework: FastAPI + Uvicorn

To run:
    uvicorn api.main:app --reload --port 8000

Then open browser:
    http://localhost:8000/docs  (Interactive API documentation)

Author: RGT-NSS Training Program
Week: 7 - Data to Deployment (MLOps Lite)
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
import numpy as np
import joblib
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI application
app = FastAPI(
    title="Churn Prediction API",
    description="Machine Learning API for predicting customer churn",
    version="1.0.0",
    docs_url="/docs",      # Swagger UI
    redoc_url="/redoc"     # ReDoc alternative
)

# Model configuration
MODEL_VERSION = "1.0.0"
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 
                          'week-06-supervised-ml-2', 'models', 
                          'tuned_random_forest.pkl')

# Global variable for model
model = None


# =============================================================================
# Pydantic Models for Request/Response Validation
# =============================================================================

class CustomerData(BaseModel):
    """
    Input model for single customer prediction.
    
    All fields are optional with default values to make testing easier.
    In production, you would make required fields mandatory.
    """
    gender: str = Field(default="Male", description="Customer gender (Male/Female)")
    SeniorCitizen: int = Field(default=0, description="Senior citizen flag (0/1)")
    Partner: str = Field(default="No", description="Has partner (Yes/No)")
    Dependents: str = Field(default="No", description="Has dependents (Yes/No)")
    tenure: int = Field(default=1, description="Number of months as customer")
    PhoneService: str = Field(default="Yes", description="Has phone service (Yes/No)")
    MultipleLines: str = Field(default="No", description="Multiple phone lines")
    InternetService: str = Field(default="DSL", description="Internet service type")
    OnlineSecurity: str = Field(default="No", description="Online security add-on")
    OnlineBackup: str = Field(default="No", description="Online backup add-on")
    DeviceProtection: str = Field(default="No", description="Device protection add-on")
    TechSupport: str = Field(default="No", description="Tech support add-on")
    StreamingTV: str = Field(default="No", description="TV streaming add-on")
    StreamingMovies: str = Field(default="No", description="Movie streaming add-on")
    Contract: str = Field(default="Month-to-month", description="Contract type")
    PaperlessBilling: str = Field(default="Yes", description="Paperless billing")
    PaymentMethod: str = Field(default="Electronic check", description="Payment method")
    MonthlyCharges: float = Field(default=29.85, description="Monthly charges in USD")
    TotalCharges: float = Field(default=29.85, description="Total charges in USD")
    
    class Config:
        json_schema_extra = {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": 29.85
            }
        }


class PredictionResponse(BaseModel):
    """Output model for prediction results."""
    churn_prediction: int = Field(..., description="0 = No Churn, 1 = Churn")
    churn_probability: float = Field(..., description="Probability of churn (0-1)")
    model_version: str = Field(..., description="Model version used")
    timestamp: str = Field(..., description="Prediction timestamp")


class BatchPredictionRequest(BaseModel):
    """Input model for batch predictions."""
    customers: List[CustomerData]
    
    class Config:
        json_schema_extra = {
            "example": {
                "customers": [
                    {
                        "gender": "Female",
                        "tenure": 1,
                        "Contract": "Month-to-month",
                        "MonthlyCharges": 29.85,
                        "TotalCharges": 29.85
                    }
                ]
            }
        }


class BatchPredictionResponse(BaseModel):
    """Output model for batch prediction results."""
    predictions: List[PredictionResponse]
    total_customers: int


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    model_loaded: bool
    model_version: str
    timestamp: str


# =============================================================================
# Startup Event - Load Model
# =============================================================================

@app.on_event("startup")
def load_model():
    """
    Load the trained model when the API starts.
    
    This runs once when the server starts, not on every request.
    """
    global model
    
    logger.info("Loading model...")
    
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            logger.info(f"[OK] Model loaded from {MODEL_PATH}")
        else:
            logger.error(f"[ERROR] Model file not found: {MODEL_PATH}")
            model = None
    except Exception as e:
        logger.error(f"[ERROR] Failed to load model: {e}")
        model = None


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/", tags=["Root"])
def root():
    """
    Root endpoint - provides basic API information.
    
    Returns:
        dict: API information and documentation link
    """
    return {
        "message": "Churn Prediction API",
        "version": MODEL_VERSION,
        "documentation": "/docs",
        "health_check": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """
    Health check endpoint.
    
    Used by monitoring systems and load balancers to verify
    the API is running and the model is loaded.
    
    Returns:
        HealthResponse: Status information
    """
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_version=MODEL_VERSION,
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
def predict(customer: CustomerData):
    """
    Make churn prediction for a single customer.
    
    Args:
        customer: CustomerData object with all features
    
    Returns:
        PredictionResponse with churn prediction and probability
    
    Raises:
        HTTPException: If model not loaded or prediction fails
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please try again later."
        )
    
    try:
        # Convert Pydantic model to DataFrame
        # The model expects a DataFrame with the same structure as training data
        input_data = pd.DataFrame([customer.model_dump()])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0, 1]
        
        logger.info(f"Prediction made: churn={prediction}, prob={probability:.4f}")
        
        return PredictionResponse(
            churn_prediction=int(prediction),
            churn_probability=float(probability),
            model_version=MODEL_VERSION,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predictions"])
def predict_batch(request: BatchPredictionRequest):
    """
    Make churn predictions for multiple customers.
    
    Args:
        request: BatchPredictionRequest containing list of customers
    
    Returns:
        BatchPredictionResponse with predictions for all customers
    
    Raises:
        HTTPException: If model not loaded or prediction fails
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please try again later."
        )
    
    try:
        # Handle empty list
        if not request.customers:
            return BatchPredictionResponse(
                predictions=[],
                total_customers=0
            )
        
        # Convert list of customers to DataFrame
        customers_data = [c.model_dump() for c in request.customers]
        input_data = pd.DataFrame(customers_data)
        
        # Make predictions
        predictions = model.predict(input_data)
        probabilities = model.predict_proba(input_data)[:, 1]
        
        # Build response list
        results = []
        for pred, prob in zip(predictions, probabilities):
            results.append(PredictionResponse(
                churn_prediction=int(pred),
                churn_probability=float(prob),
                model_version=MODEL_VERSION,
                timestamp=datetime.now().isoformat()
            ))
        
        logger.info(f"Batch prediction: {len(results)} customers processed")
        
        return BatchPredictionResponse(
            predictions=results,
            total_customers=len(results)
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 70)
    print("Week 7: Churn Prediction API")
    print("=" * 70)
    print("\nStarting server...")
    print("API documentation available at: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop")
    print("=" * 70)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
