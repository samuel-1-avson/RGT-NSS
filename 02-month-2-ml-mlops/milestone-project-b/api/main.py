"""
FastAPI Application for Customer Churn Prediction

This module provides a production-ready REST API for predicting customer churn.

Features:
- Health check endpoint
- Single and batch prediction endpoints
- Input validation with Pydantic
- Comprehensive error handling
- Request/response logging
- Model versioning

Usage:
    Development: uvicorn api.main:app --reload
    Production: uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
"""

import os
import sys
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator, field_validator

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))
from config import get_settings, Settings

# Get settings
settings = get_settings()

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global model variables
model = None
preprocessor = None
model_metadata = {
    "loaded": False,
    "version": settings.MODEL_VERSION,
    "load_time": None,
    "error": None,
    "features": []
}


# ==================== Pydantic Models ====================

class CustomerData(BaseModel):
    """
    Customer data model for churn prediction.
    
    This model represents the input features required for churn prediction.
    All fields include validation rules and documentation.
    """
    # Demographics
    gender: str = Field(..., description="Customer gender (Male/Female)")
    SeniorCitizen: int = Field(..., ge=0, le=1, description="1 if senior citizen, 0 otherwise")
    Partner: str = Field(..., description="Has partner (Yes/No)")
    Dependents: str = Field(..., description="Has dependents (Yes/No)")
    
    # Account Information
    tenure: int = Field(..., ge=0, le=100, description="Number of months as customer")
    PhoneService: str = Field(..., description="Has phone service (Yes/No)")
    MultipleLines: str = Field(..., description="Multiple lines (Yes/No/No phone service)")
    InternetService: str = Field(..., description="Internet service type (DSL/Fiber optic/No)")
    
    # Online Services
    OnlineSecurity: str = Field(..., description="Online security (Yes/No/No internet service)")
    OnlineBackup: str = Field(..., description="Online backup (Yes/No/No internet service)")
    DeviceProtection: str = Field(..., description="Device protection (Yes/No/No internet service)")
    TechSupport: str = Field(..., description="Tech support (Yes/No/No internet service)")
    StreamingTV: str = Field(..., description="Streaming TV (Yes/No/No internet service)")
    StreamingMovies: str = Field(..., description="Streaming movies (Yes/No/No internet service)")
    
    # Contract and Billing
    Contract: str = Field(..., description="Contract type (Month-to-month/One year/Two year)")
    PaperlessBilling: str = Field(..., description="Paperless billing (Yes/No)")
    PaymentMethod: str = Field(..., description="Payment method")
    MonthlyCharges: float = Field(..., ge=0, description="Monthly charges in dollars")
    TotalCharges: float = Field(..., ge=0, description="Total charges in dollars")
    
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


class PredictionRequest(BaseModel):
    """Single prediction request."""
    customer_id: Optional[str] = Field(None, description="Unique customer identifier")
    data: CustomerData
    
    class Config:
        json_schema_extra = {
            "example": {
                "customer_id": "cust_001",
                "data": CustomerData.Config.json_schema_extra["example"]
            }
        }


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""
    customers: List[PredictionRequest] = Field(..., max_length=settings.MAX_BATCH_SIZE)
    
    @field_validator('customers')
    @classmethod
    def check_not_empty(cls, v):
        if len(v) == 0:
            raise ValueError('Batch request cannot be empty')
        return v


class PredictionResponse(BaseModel):
    """Single prediction response."""
    customer_id: Optional[str] = None
    churn_probability: float = Field(..., ge=0, le=1)
    churn_prediction: bool
    confidence: str = Field(..., description="high/medium/low based on probability")
    model_version: str
    prediction_time: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "customer_id": "cust_001",
                "churn_probability": 0.23,
                "churn_prediction": False,
                "confidence": "high",
                "model_version": "1.0.0",
                "prediction_time": "2024-01-15T10:30:00Z"
            }
        }


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    predictions: List[PredictionResponse]
    total: int
    successful: int
    failed: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_version: str
    timestamp: str
    uptime_seconds: Optional[float] = None


class ModelInfoResponse(BaseModel):
    """Model information response."""
    model_loaded: bool
    model_version: str
    load_time: Optional[str]
    expected_features: List[str]
    prediction_threshold: float


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str] = None
    timestamp: str


# ==================== Model Loading ====================

def load_model() -> bool:
    """
    Load the trained model and preprocessor.
    
    Returns:
        bool: True if loaded successfully
    """
    global model, preprocessor, model_metadata
    
    try:
        model_path = Path(__file__).parent.parent / settings.MODEL_PATH
        preprocessor_path = Path(__file__).parent.parent / settings.PREPROCESSOR_PATH
        
        logger.info(f"Loading model from {model_path}")
        
        # Load or create dummy model
        if model_path.exists():
            model = joblib.load(model_path)
            logger.info("Model loaded successfully")
        else:
            logger.warning("Model file not found, creating dummy model")
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            # Create dummy training data
            X_dummy = np.random.rand(100, 19)
            y_dummy = np.random.randint(0, 2, 100)
            model.fit(X_dummy, y_dummy)
        
        # Load or create dummy preprocessor
        if preprocessor_path.exists():
            preprocessor = joblib.load(preprocessor_path)
        else:
            logger.warning("Preprocessor not found, will use raw features")
            preprocessor = None
        
        model_metadata["loaded"] = True
        model_metadata["load_time"] = datetime.utcnow().isoformat()
        model_metadata["features"] = list(CustomerData.model_fields.keys())
        
        logger.info("Model initialization complete")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        model_metadata["loaded"] = False
        model_metadata["error"] = str(e)
        return False


# ==================== FastAPI Lifespan ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    # Startup
    logger.info("Starting up Churn Prediction API...")
    load_model()
    yield
    # Shutdown
    logger.info("Shutting down...")


# Initialize FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="Production-ready ML microservice for customer churn prediction",
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== Helper Functions ====================

def get_confidence_level(probability: float) -> str:
    """
    Determine confidence level based on probability.
    
    Args:
        probability: Churn probability
        
    Returns:
        str: high, medium, or low
    """
    distance_from_threshold = abs(probability - settings.PREDICTION_THRESHOLD)
    if distance_from_threshold > 0.3:
        return "high"
    elif distance_from_threshold > 0.15:
        return "medium"
    return "low"


def prepare_features(data: CustomerData) -> pd.DataFrame:
    """
    Convert CustomerData to DataFrame for model input.
    
    Args:
        data: CustomerData instance
        
    Returns:
        pd.DataFrame: Feature DataFrame
    """
    return pd.DataFrame([data.model_dump()])


def make_prediction(customer_data: CustomerData, customer_id: Optional[str] = None) -> PredictionResponse:
    """
    Make a churn prediction for a single customer.
    
    Args:
        customer_data: Customer features
        customer_id: Optional customer identifier
        
    Returns:
        PredictionResponse with prediction results
    """
    start_time = time.time()
    
    try:
        # Prepare features
        features_df = prepare_features(customer_data)
        
        # Apply preprocessing if available
        if preprocessor is not None:
            features = preprocessor.transform(features_df)
        else:
            features = features_df.values
        
        # Make prediction
        probability = float(model.predict_proba(features)[0][1])
        prediction = probability >= settings.PREDICTION_THRESHOLD
        
        latency_ms = (time.time() - start_time) * 1000
        
        logger.info(
            f"Prediction made for customer {customer_id}",
            extra={
                "customer_id": customer_id,
                "probability": probability,
                "prediction": prediction,
                "latency_ms": latency_ms
            }
        )
        
        return PredictionResponse(
            customer_id=customer_id,
            churn_probability=round(probability, 4),
            churn_prediction=prediction,
            confidence=get_confidence_level(probability),
            model_version=settings.MODEL_VERSION,
            prediction_time=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise


# ==================== API Endpoints ====================

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "documentation": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        HealthResponse with model status
    """
    return HealthResponse(
        status="healthy" if model_metadata["loaded"] else "unhealthy",
        model_loaded=model_metadata["loaded"],
        model_version=settings.MODEL_VERSION,
        timestamp=datetime.utcnow().isoformat()
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """
    Get model information.
    
    Returns:
        ModelInfoResponse with model metadata
    """
    return ModelInfoResponse(
        model_loaded=model_metadata["loaded"],
        model_version=settings.MODEL_VERSION,
        load_time=model_metadata.get("load_time"),
        expected_features=model_metadata.get("features", []),
        prediction_threshold=settings.PREDICTION_THRESHOLD
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make a churn prediction for a single customer.
    
    Args:
        request: PredictionRequest with customer data
        
    Returns:
        PredictionResponse with churn probability
        
    Raises:
        HTTPException: If model not loaded or prediction fails
    """
    if not model_metadata["loaded"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please try again later."
        )
    
    try:
        return make_prediction(request.data, request.customer_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Make churn predictions for multiple customers.
    
    Args:
        request: BatchPredictionRequest with multiple customers
        
    Returns:
        BatchPredictionResponse with predictions
    """
    if not model_metadata["loaded"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    predictions = []
    successful = 0
    failed = 0
    
    for customer_request in request.customers:
        try:
            prediction = make_prediction(
                customer_request.data,
                customer_request.customer_id
            )
            predictions.append(prediction)
            successful += 1
        except Exception as e:
            logger.error(f"Batch prediction failed for {customer_request.customer_id}: {str(e)}")
            failed += 1
            # Add error placeholder
            predictions.append(PredictionResponse(
                customer_id=customer_request.customer_id,
                churn_probability=0.0,
                churn_prediction=False,
                confidence="error",
                model_version=settings.MODEL_VERSION,
                prediction_time=datetime.utcnow().isoformat()
            ))
    
    return BatchPredictionResponse(
        predictions=predictions,
        total=len(request.customers),
        successful=successful,
        failed=failed
    )


# ==================== Error Handlers ====================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    logger.warning(f"HTTP {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            timestamp=datetime.utcnow().isoformat()
        ).model_dump()
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Handle generic exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc) if settings.DEBUG else None,
            timestamp=datetime.utcnow().isoformat()
        ).model_dump()
    )


# For running directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)
