"""
Common Pydantic schemas used across the API.
"""

from typing import Optional, List, Dict, Any, Generic, TypeVar
from pydantic import BaseModel, Field
from datetime import datetime


T = TypeVar('T')


class ErrorDetail(BaseModel):
    """Detailed error information."""
    field: Optional[str] = None
    message: str
    value: Optional[Any] = None
    constraint: Optional[str] = None


class ErrorResponse(BaseModel):
    """Standard error response."""
    success: bool = False
    error: str
    message: str
    details: Optional[List[ErrorDetail]] = None
    error_code: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    request_id: Optional[str] = None
    documentation_url: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "error": "Model not found",
                "message": "Model with ID 'abc123' does not exist",
                "error_code": "MODEL_NOT_FOUND",
                "timestamp": "2024-01-01T00:00:00"
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., pattern='^(healthy|unhealthy|degraded)$')
    version: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    components: Dict[str, str]
    uptime_seconds: Optional[float] = None
    environment: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "timestamp": "2024-01-01T00:00:00",
                "components": {
                    "api": "up",
                    "models": "ready",
                    "training": "ready",
                    "storage": "connected"
                }
            }
        }


class StatusResponse(BaseModel):
    """API status response."""
    version: str
    environment: str
    active_models: int
    active_training_sessions: int
    total_requests: Optional[int] = None
    uptime_seconds: Optional[float] = None
    endpoints: Dict[str, int]
    rate_limits: Optional[Dict[str, Any]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "version": "1.0.0",
                "environment": "production",
                "active_models": 5,
                "active_training_sessions": 2,
                "endpoints": {
                    "total": 25,
                    "documented": 20
                }
            }
        }


class PaginatedResponse(BaseModel, Generic[T]):
    """Generic paginated response."""
    success: bool = True
    data: List[T]
    total: int
    page: int = Field(..., ge=1)
    per_page: int = Field(..., ge=1, le=100)
    total_pages: int
    has_next: bool
    has_prev: bool
    next_page: Optional[int] = None
    prev_page: Optional[int] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "data": [],
                "total": 100,
                "page": 1,
                "per_page": 20,
                "total_pages": 5,
                "has_next": True,
                "has_prev": False
            }
        }


class SuccessResponse(BaseModel):
    """Generic success response."""
    success: bool = True
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Operation completed successfully",
                "data": {},
                "timestamp": "2024-01-01T00:00:00"
            }
        }


class GPUStatusResponse(BaseModel):
    """GPU status response."""
    pytorch_available: bool
    cuda_available: bool
    device: str
    pytorch_version: Optional[str] = None
    gpu_name: Optional[str] = None
    gpu_total_memory_mb: Optional[float] = None
    gpu_allocated_memory_mb: Optional[float] = None
    gpu_reserved_memory_mb: Optional[float] = None
    gpu_utilization_percent: Optional[float] = None
    cuda_version: Optional[str] = None
    cudnn_version: Optional[str] = None
    message: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "pytorch_available": True,
                "cuda_available": True,
                "device": "cuda:0",
                "gpu_name": "NVIDIA RTX 4090",
                "gpu_total_memory_mb": 24576
            }
        }


class WebSocketMessage(BaseModel):
    """WebSocket message format."""
    type: str = Field(..., pattern='^(status|metrics|error|complete|stopped)$')
    data: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    
    class Config:
        json_schema_extra = {
            "example": {
                "type": "metrics",
                "data": {"loss": 2.5, "step": 100},
                "timestamp": "2024-01-01T00:00:00"
            }
        }
