"""Pydantic schemas for request/response validation."""

from .models import (
    GPTConfigRequest,
    ModelCreateRequest,
    ModelResponse,
    ModelListResponse,
    ModelInfo,
)
from .training import (
    TrainingConfigRequest,
    TrainingStartRequest,
    TrainingStatusResponse,
    TrainingMetricsResponse,
    TrainingHistoryResponse,
)
from .inference import (
    GenerationRequest,
    GenerationResponse,
    TokenizeRequest,
    TokenizeResponse,
    ForwardPassRequest,
    ForwardPassResponse,
)
from .common import (
    ErrorResponse,
    HealthResponse,
    StatusResponse,
    PaginatedResponse,
)

__all__ = [
    # Model schemas
    "GPTConfigRequest",
    "ModelCreateRequest", 
    "ModelResponse",
    "ModelListResponse",
    "ModelInfo",
    # Training schemas
    "TrainingConfigRequest",
    "TrainingStartRequest",
    "TrainingStatusResponse",
    "TrainingMetricsResponse",
    "TrainingHistoryResponse",
    # Inference schemas
    "GenerationRequest",
    "GenerationResponse",
    "TokenizeRequest",
    "TokenizeResponse",
    "ForwardPassRequest",
    "ForwardPassResponse",
    # Common schemas
    "ErrorResponse",
    "HealthResponse",
    "StatusResponse",
    "PaginatedResponse",
]
