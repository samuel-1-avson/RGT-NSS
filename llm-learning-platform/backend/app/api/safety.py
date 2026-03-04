"""
AI Safety & Alignment API Router
"""

from typing import List, Optional
from fastapi import APIRouter
from pydantic import BaseModel

from app.core.safety import (
    evaluate_safety, run_redteam_suite, apply_constitutional_ai,
    get_safety_categories, get_constitutional_principles,
)

router = APIRouter()


class SafetyEvalRequest(BaseModel):
    text: str = "This is a test response for safety evaluation."


class ConstitutionalRequest(BaseModel):
    response: str = "Here is a model response to evaluate."
    principle_indices: Optional[List[int]] = None


@router.post("/evaluate")
async def evaluate(req: SafetyEvalRequest):
    """Evaluate text for safety concerns."""
    return evaluate_safety(req.text)


@router.post("/redteam")
async def run_redteam():
    """Run automated red-team test suite."""
    return run_redteam_suite()


@router.post("/constitutional")
async def apply_constitutional(req: ConstitutionalRequest):
    """Apply Constitutional AI principles to evaluate a response."""
    return apply_constitutional_ai(req.response, req.principle_indices)


@router.get("/categories")
async def get_categories():
    """List safety risk categories."""
    return {"categories": get_safety_categories()}


@router.get("/principles")
async def get_principles():
    """List Constitutional AI principles."""
    return {"principles": get_constitutional_principles()}
