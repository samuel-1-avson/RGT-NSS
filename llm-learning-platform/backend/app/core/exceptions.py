"""
Custom exceptions for the LLM Learning Platform.
Provides structured error handling across the application.
"""

from fastapi import HTTPException, status
from typing import Optional, Dict, Any


class BaseAppException(Exception):
    """Base exception for all application errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


# =============================================================================
# Model Exceptions
# =============================================================================

class ModelNotFoundError(HTTPException):
    """Raised when a requested model is not found."""
    
    def __init__(self, model_id: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "Model not found",
                "model_id": model_id,
                "message": f"Model with ID '{model_id}' does not exist or has been deleted."
            }
        )


class ModelCreationError(HTTPException):
    """Raised when model creation fails."""
    
    def __init__(self, message: str, config: Optional[Dict] = None):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "Model creation failed",
                "message": message,
                "config": config
            }
        )


class InvalidModelConfigError(HTTPException):
    """Raised when model configuration is invalid."""
    
    def __init__(self, field: str, value: Any, constraint: str):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "Invalid model configuration",
                "field": field,
                "value": value,
                "constraint": constraint,
                "message": f"Field '{field}' with value '{value}' violates constraint: {constraint}"
            }
        )


# =============================================================================
# Training Exceptions
# =============================================================================

class TrainingSessionNotFoundError(HTTPException):
    """Raised when a training session is not found."""
    
    def __init__(self, session_id: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "Training session not found",
                "session_id": session_id,
                "message": f"Training session '{session_id}' does not exist."
            }
        )


class TrainingError(HTTPException):
    """Raised when training operation fails."""
    
    def __init__(self, message: str, session_id: Optional[str] = None):
        detail = {
            "error": "Training error",
            "message": message
        }
        if session_id:
            detail["session_id"] = session_id
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail
        )


class TrainingAlreadyRunningError(HTTPException):
    """Raised when attempting to start training that's already running."""
    
    def __init__(self, session_id: str):
        super().__init__(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error": "Training already running",
                "session_id": session_id,
                "message": f"Training session '{session_id}' is already active."
            }
        )


# =============================================================================
# Inference Exceptions
# =============================================================================

class InferenceError(HTTPException):
    """Raised when inference operation fails."""
    
    def __init__(self, message: str, model_id: Optional[str] = None):
        detail = {
            "error": "Inference error",
            "message": message
        }
        if model_id:
            detail["model_id"] = model_id
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail
        )


class TokenizationError(HTTPException):
    """Raised when tokenization fails."""
    
    def __init__(self, message: str, text: Optional[str] = None):
        detail = {
            "error": "Tokenization error",
            "message": message
        }
        if text:
            detail["text_preview"] = text[:100] + "..." if len(text) > 100 else text
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail
        )


# =============================================================================
# Storage Exceptions
# =============================================================================

class StorageError(HTTPException):
    """Raised when storage operation fails."""
    
    def __init__(self, message: str, operation: Optional[str] = None):
        detail = {
            "error": "Storage error",
            "message": message
        }
        if operation:
            detail["operation"] = operation
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail
        )


class CacheError(HTTPException):
    """Raised when cache operation fails."""
    
    def __init__(self, message: str):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Cache error",
                "message": message
            }
        )


# =============================================================================
# Authentication/Authorization Exceptions
# =============================================================================

class AuthenticationError(HTTPException):
    """Raised when authentication fails."""
    
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "Authentication error",
                "message": message
            },
            headers={"WWW-Authenticate": "Bearer"}
        )


class AuthorizationError(HTTPException):
    """Raised when user lacks required permissions."""
    
    def __init__(self, message: str = "Insufficient permissions"):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "Authorization error",
                "message": message
            }
        )


class RateLimitExceededError(HTTPException):
    """Raised when rate limit is exceeded."""
    
    def __init__(self, retry_after: int = 60):
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": "Rate limit exceeded",
                "message": f"Too many requests. Please retry after {retry_after} seconds.",
                "retry_after": retry_after
            },
            headers={"Retry-After": str(retry_after)}
        )


# =============================================================================
# Validation Exceptions
# =============================================================================

class ValidationError(HTTPException):
    """Raised when input validation fails."""
    
    def __init__(self, field: str, message: str, value: Optional[Any] = None):
        detail = {
            "error": "Validation error",
            "field": field,
            "message": message
        }
        if value is not None:
            detail["value"] = str(value)
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=detail
        )


# =============================================================================
# GPU/Hardware Exceptions
# =============================================================================

class GPUError(HTTPException):
    """Raised when GPU operation fails."""
    
    def __init__(self, message: str, device_id: Optional[int] = None):
        detail = {
            "error": "GPU error",
            "message": message
        }
        if device_id is not None:
            detail["device_id"] = device_id
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail
        )


class OutOfMemoryError(HTTPException):
    """Raised when out of memory (CPU or GPU)."""
    
    def __init__(self, message: str = "Out of memory", device: Optional[str] = None):
        detail = {
            "error": "Out of memory",
            "message": message
        }
        if device:
            detail["device"] = device
        super().__init__(
            status_code=status.HTTP_507_INSUFFICIENT_STORAGE,
            detail=detail
        )


# =============================================================================
# WebSocket Exceptions
# =============================================================================

class WebSocketError(Exception):
    """Raised for WebSocket-related errors (not HTTP)."""
    
    def __init__(self, message: str, code: int = 4000):
        self.message = message
        self.code = code
        super().__init__(self.message)


class ConnectionError(WebSocketError):
    """Raised when WebSocket connection fails."""
    
    def __init__(self, message: str = "Connection error"):
        super().__init__(message, code=4001)
