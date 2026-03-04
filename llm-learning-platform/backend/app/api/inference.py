"""
Inference API

Text generation with multiple sampling strategies,
forward pass inspection, and step-by-step analysis.
"""

from typing import List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.model import MicroGPT, GPTConfig, PRESET_CONFIGS

router = APIRouter()

# Shared model store (references same as models.py in production)
from app.api.models import _models


class GenerateRequest(BaseModel):
    model_id: Optional[str] = None
    prompt_ids: List[int] = Field(default=[2])  # BOS token
    max_new_tokens: int = Field(default=50, le=500)
    temperature: float = Field(default=1.0, ge=0.01, le=5.0)
    top_k: int = Field(default=0, ge=0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)


class ForwardPassRequest(BaseModel):
    model_id: Optional[str] = None
    token_ids: List[int] = Field(default=[2, 10, 20, 30])
    store_intermediates: bool = True


@router.post("/generate")
async def generate_text(request: GenerateRequest):
    """Generate text autoregressively with sampling strategies."""
    if request.model_id and request.model_id in _models:
        model = _models[request.model_id]
    else:
        model = MicroGPT(PRESET_CONFIGS["nano"])

    prompt = np.array(request.prompt_ids)
    generated, step_metadata = model.generate(
        prompt,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_k=request.top_k,
        top_p=request.top_p,
    )

    return {
        "generated_ids": generated[0].tolist(),
        "num_tokens_generated": request.max_new_tokens,
        "steps": step_metadata[:10],  # first 10 steps for visualization
        "settings": {
            "temperature": request.temperature,
            "top_k": request.top_k,
            "top_p": request.top_p,
        },
    }


@router.post("/forward")
async def forward_pass(request: ForwardPassRequest):
    """Run a single forward pass with intermediate inspection."""
    if request.model_id and request.model_id in _models:
        model = _models[request.model_id]
    else:
        model = MicroGPT(PRESET_CONFIGS["nano"])

    token_ids = np.array([request.token_ids])
    result = model.forward(
        token_ids,
        store_intermediates=request.store_intermediates,
    )

    logits = result["logits"]
    top_k_ids = np.argsort(logits[0, -1])[-10:][::-1]

    response = {
        "logits_shape": list(logits.shape),
        "next_token_predictions": [
            {"token_id": int(tid), "logit": float(logits[0, -1, tid])}
            for tid in top_k_ids
        ],
    }

    if request.store_intermediates and "intermediates" in result:
        intermediates = result["intermediates"]
        response["layer_shapes"] = {
            k: list(v.shape) if isinstance(v, np.ndarray) else str(type(v))
            for k, v in intermediates.items()
        }

    return response
