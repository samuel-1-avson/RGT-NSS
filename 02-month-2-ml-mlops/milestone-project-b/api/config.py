"""
Configuration settings for the Churn Prediction API.

This module centralizes all configuration settings using Pydantic Settings,
enabling easy environment-based configuration.
"""

import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    Environment variables are automatically loaded from .env file if present.
    """
    
    # API Configuration
    APP_NAME: str = Field(default="Customer Churn Prediction API")
    APP_VERSION: str = Field(default="1.0.0")
    DEBUG: bool = Field(default=False)
    
    # Server Configuration
    HOST: str = Field(default="0.0.0.0")
    PORT: int = Field(default=8000)
    WORKERS: int = Field(default=1)
    
    # Model Configuration
    MODEL_PATH: str = Field(default="models/churn_model.pkl")
    PREPROCESSOR_PATH: str = Field(default="models/preprocessor.pkl")
    MODEL_VERSION: str = Field(default="1.0.0")
    
    # Logging Configuration
    LOG_LEVEL: str = Field(default="INFO")
    LOG_FORMAT: str = Field(default="json")  # json or text
    
    # Monitoring Configuration
    ENABLE_METRICS: bool = Field(default=True)
    METRICS_PORT: int = Field(default=9090)
    
    # Security Configuration
    API_KEY: Optional[str] = Field(default=None)
    ALLOWED_ORIGINS: List[str] = Field(default=["*"])
    
    # Prediction Configuration
    PREDICTION_THRESHOLD: float = Field(default=0.5)
    MAX_BATCH_SIZE: int = Field(default=100)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Create global settings instance
settings = Settings()


def get_settings() -> Settings:
    """
    Get application settings.
    
    Returns:
        Settings: Application configuration
    """
    return settings
