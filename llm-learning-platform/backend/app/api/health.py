"""Health check endpoints with GPU/device info."""

from typing import Dict, Optional
from pydantic import BaseModel
from fastapi import APIRouter

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class ReadinessResponse(BaseModel):
    status: str
    device: str
    cuda_available: bool
    gpu_name: Optional[str] = None
    torch_version: str


@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        service="llm-learning-platform",
        version="4.0.0",
    )


@router.get("/health/ready", response_model=ReadinessResponse)
async def readiness_check():
    from app.core.device import get_device_info
    info = get_device_info()
    return ReadinessResponse(
        status="ready",
        device=info["device"],
        cuda_available=info["cuda_available"],
        gpu_name=info.get("gpu_name"),
        torch_version=info["torch_version"],
    )


@router.get("/health/gpu")
async def gpu_info():
    """Detailed GPU information."""
    from app.core.device import get_device_info
    return get_device_info()
