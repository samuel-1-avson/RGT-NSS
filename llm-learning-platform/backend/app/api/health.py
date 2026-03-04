"""Health check endpoints."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "llm-learning-platform",
        "version": "3.0.0",
    }


@router.get("/health/ready")
async def readiness_check():
    return {"status": "ready"}
