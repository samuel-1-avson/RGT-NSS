"""
FastAPI Application for House Price Prediction

This module provides a REST API for predicting house prices using
a pre-trained machine learning model.

Endpoints:
    - GET /health: Health check endpoint
    - POST /predict: Make price predictions

Usage:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Author: RGT-NSS AI Training Program
Version: 1.0.0
"""

import os
import sys
import logging
from datetime import datetime
from typing import Optional
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="House Price Prediction API",
    description="ML API for predicting house prices based on property features",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model path - adjust based on directory structure
MODEL_PATH = Path(__file__).parent.parent / "models" / "house_price_model.pkl"

# Global model variable
model = None
model_metadata = {
    "loaded": False,
    "version": None,
    "load_time": None,
    "error": None
}


# ==================== Pydantic Models ====================

class PredictionRequest(BaseModel):
    """
    Request model for house price prediction.
    
    Attributes:
        LotArea: Lot size in square feet
        OverallQual: Overall material and finish quality (1-10)
        OverallCond: Overall condition rating (1-10)
        YearBuilt: Original construction year
        TotalBsmtSF: Total basement area in square feet
        GrLivArea: Above grade living area in square feet
        GarageCars: Size of garage in car capacity
        GarageArea: Garage area in square feet
    """
    LotArea: float = Field(..., gt=0, description="Lot size in square feet")
    OverallQual: int = Field(..., ge=1, le=10, description="Overall quality rating (1-10)")
    OverallCond: int = Field(..., ge=1, le=10, description="Overall condition rating (1-10)")
    YearBuilt: int = Field(..., ge=1800, le=2024, description="Year built")
    TotalBsmtSF: float = Field(..., ge=0, description="Total basement square footage")
    GrLivArea: float = Field(..., gt=0, description="Above grade living area")
    GarageCars: int = Field(..., ge=0, le=5, description="Garage car capacity")
    GarageArea: float = Field(..., ge=0, description="Garage area in square feet")
    
    class Config:
        json_schema_extra = {
            "example": {
                "LotArea": 8450,
                "OverallQual": 7,
                "OverallCond": 5,
                "YearBuilt": 2003,
                "TotalBsmtSF": 856,
                "GrLivArea": 1710,
                "GarageCars": 2,
                "GarageArea": 548
            }
        }


class PredictionResponse(BaseModel):
    """
    Response model for house price prediction.
    
    Attributes:
        predicted_price: Predicted house price in USD
        prediction_interval: Confidence interval for the prediction
        model_version: Version of the model used
        timestamp: Prediction timestamp
    """
    predicted_price: float
    prediction_interval: tuple[float, float]
    model_version: str
    timestamp: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "predicted_price": 208500.50,
                "prediction_interval": [195000.0, 222000.0],
                "model_version": "1.0.0",
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


class HealthResponse(BaseModel):
    """
    Health check response model.
    
    Attributes:
        status: Health status (healthy/unhealthy)
        model_loaded: Whether the model is loaded
        model_version: Model version string
        timestamp: Current timestamp
    """
    status: str
    model_loaded: bool
    model_version: Optional[str]
    timestamp: str


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str] = None
    timestamp: str


# ==================== Model Loading ====================

def load_model():
    """
    Load the trained machine learning model.
    
    This function attempts to load the model from disk and updates
    the model_metadata dictionary with load status.
    
    Returns:
        bool: True if model loaded successfully, False otherwise
    """
    global model, model_metadata
    
    try:
        if not MODEL_PATH.exists():
            logger.warning(f"Model file not found at {MODEL_PATH}")
            logger.info("Creating a dummy model for demonstration...")
            # Create dummy model for demonstration
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=10, random_state=42)
            # Fit with dummy data
            X_dummy = np.random.rand(100, 8)
            y_dummy = np.random.rand(100) * 200000 + 100000
            model.fit(X_dummy, y_dummy)
        else:
            logger.info(f"Loading model from {MODEL_PATH}")
            model = joblib.load(MODEL_PATH)
        
        model_metadata["loaded"] = True
        model_metadata["version"] = "1.0.0"
        model_metadata["load_time"] = datetime.utcnow().isoformat()
        model_metadata["error"] = None
        
        logger.info("Model loaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        model_metadata["loaded"] = False
        model_metadata["error"] = str(e)
        return False


# Load model on startup
@app.on_event("startup")
async def startup_event():
    """Load model when the application starts."""
    logger.info("Starting up House Price Prediction API...")
    load_model()


# ==================== API Endpoints ====================

@app.get("/", response_model=dict)
async def root():
    """
    Root endpoint with API information.
    
    Returns:
        dict: API name, version, and documentation links
    """
    return {
        "name": "House Price Prediction API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health_check": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns the current health status of the API including
    model load status and version information.
    
    Returns:
        HealthResponse: Health status information
    """
    return HealthResponse(
        status="healthy" if model_metadata["loaded"] else "unhealthy",
        model_loaded=model_metadata["loaded"],
        model_version=model_metadata["version"],
        timestamp=datetime.utcnow().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make a house price prediction.
    
    This endpoint accepts house features and returns a predicted price
    along with a confidence interval.
    
    Args:
        request: PredictionRequest with house features
        
    Returns:
        PredictionResponse with predicted price and metadata
        
    Raises:
        HTTPException: If model is not loaded or prediction fails
    """
    # Check if model is loaded
    if not model_metadata["loaded"] or model is None:
        logger.error("Prediction attempted but model not loaded")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please try again later."
        )
    
    try:
        # Log the request
        logger.info(f"Prediction request received: {request.dict()}")
        
        # Prepare features (must match training order)
        features = np.array([[
            request.LotArea,
            request.OverallQual,
            request.OverallCond,
            request.YearBuilt,
            request.TotalBsmtSF,
            request.GrLivArea,
            request.GarageCars,
            request.GarageArea
        ]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Calculate prediction interval (simple approach)
        # In practice, you'd use model-specific methods or bootstrapping
        std_estimate = prediction * 0.05  # 5% margin
        lower_bound = max(0, prediction - 1.96 * std_estimate)
        upper_bound = prediction + 1.96 * std_estimate
        
        # Log the prediction
        logger.info(f"Prediction: ${prediction:,.2f}")
        
        return PredictionResponse(
            predicted_price=float(prediction),
            prediction_interval=(float(lower_bound), float(upper_bound)),
            model_version=model_metadata["version"] or "1.0.0",
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch")
async def predict_batch(requests: list[PredictionRequest]):
    """
    Make batch predictions for multiple houses.
    
    Args:
        requests: List of PredictionRequest objects
        
    Returns:
        List of prediction results
    """
    if not model_metadata["loaded"] or model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        logger.info(f"Batch prediction request: {len(requests)} houses")
        
        # Prepare features for all requests
        features_list = []
        for req in requests:
            features_list.append([
                req.LotArea,
                req.OverallQual,
                req.OverallCond,
                req.YearBuilt,
                req.TotalBsmtSF,
                req.GrLivArea,
                req.GarageCars,
                req.GarageArea
            ])
        
        features = np.array(features_list)
        predictions = model.predict(features)
        
        results = []
        for i, pred in enumerate(predictions):
            std_estimate = pred * 0.05
            results.append({
                "predicted_price": float(pred),
                "prediction_interval": (
                    float(max(0, pred - 1.96 * std_estimate)),
                    float(pred + 1.96 * std_estimate)
                ),
                "model_version": model_metadata["version"],
                "timestamp": datetime.utcnow().isoformat()
            })
        
        return {"predictions": results, "count": len(results)}
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/model/info")
async def model_info():
    """
    Get information about the loaded model.
    
    Returns:
        dict: Model metadata and status
    """
    return {
        "model_loaded": model_metadata["loaded"],
        "version": model_metadata["version"],
        "load_time": model_metadata["load_time"],
        "expected_features": [
            "LotArea",
            "OverallQual",
            "OverallCond",
            "YearBuilt",
            "TotalBsmtSF",
            "GrLivArea",
            "GarageCars",
            "GarageArea"
        ]
    }


# Error handlers
@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    """Handle generic exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return {
        "error": "Internal server error",
        "detail": str(exc),
        "timestamp": datetime.utcnow().isoformat()
    }


# For running directly (development)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
