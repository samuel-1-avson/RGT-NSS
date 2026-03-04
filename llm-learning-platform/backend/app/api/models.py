"""
Model Management API

Create, configure, inspect, and manage GPT model instances.
"""

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.model import GPTConfig, MicroGPT, PRESET_CONFIGS

router = APIRouter()

# In-memory model store (would be DB-backed in production)
_models: dict = {}


class CreateModelRequest(BaseModel):
    preset: Optional[str] = Field(None, description="Preset name: nano, micro, mini, small, medium, large")
    vocab_size: int = 256
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 4
    d_ff: int = 512
    max_seq_len: int = 256
    dropout: float = 0.1
    norm_type: str = "rmsnorm"
    norm_placement: str = "pre"
    activation: str = "gelu"
    attention_type: str = "full"
    positional_encoding: str = "sinusoidal"
    use_bias: bool = True
    tie_weights: bool = True


class ModelInfoResponse(BaseModel):
    model_id: str
    config: dict
    num_parameters: int
    status: str


@router.post("", response_model=ModelInfoResponse)
async def create_model(request: CreateModelRequest):
    """Create a new GPT model instance."""
    if request.preset and request.preset in PRESET_CONFIGS:
        config = PRESET_CONFIGS[request.preset]
    else:
        config = GPTConfig(
            vocab_size=request.vocab_size,
            d_model=request.d_model,
            num_heads=request.num_heads,
            num_layers=request.num_layers,
            d_ff=request.d_ff,
            max_seq_len=request.max_seq_len,
            dropout=request.dropout,
            norm_type=request.norm_type,
            norm_placement=request.norm_placement,
            activation=request.activation,
            attention_type=request.attention_type,
            positional_encoding=request.positional_encoding,
            use_bias=request.use_bias,
            tie_weights=request.tie_weights,
        )

    import uuid
    model_id = str(uuid.uuid4())
    model = MicroGPT(config)
    _models[model_id] = model

    return ModelInfoResponse(
        model_id=model_id,
        config=model.get_config_summary(),
        num_parameters=model.num_parameters(),
        status="created",
    )


@router.get("/{model_id}", response_model=ModelInfoResponse)
async def get_model(model_id: str):
    """Get model information."""
    if model_id not in _models:
        raise HTTPException(status_code=404, detail="Model not found")
    model = _models[model_id]
    return ModelInfoResponse(
        model_id=model_id,
        config=model.get_config_summary(),
        num_parameters=model.num_parameters(),
        status="ready",
    )


@router.get("")
async def list_models():
    """List all model instances."""
    return [
        {
            "model_id": mid,
            "num_parameters": m.num_parameters(),
            "d_model": m.config.d_model,
            "num_layers": m.config.num_layers,
        }
        for mid, m in _models.items()
    ]


@router.delete("/{model_id}")
async def delete_model(model_id: str):
    """Delete a model instance."""
    if model_id not in _models:
        raise HTTPException(status_code=404, detail="Model not found")
    del _models[model_id]
    return {"status": "deleted", "model_id": model_id}


@router.post("/{model_id}/reset")
async def reset_model(model_id: str):
    """Reinitialize model weights."""
    if model_id not in _models:
        raise HTTPException(status_code=404, detail="Model not found")
    old_model = _models[model_id]
    _models[model_id] = MicroGPT(old_model.config)
    return {"status": "reset", "model_id": model_id}


@router.get("/presets/list")
async def list_presets():
    """List available model presets."""
    return {
        name: {
            "d_model": cfg.d_model,
            "num_heads": cfg.num_heads,
            "num_layers": cfg.num_layers,
            "d_ff": cfg.d_ff,
            "estimated_parameters": cfg.num_parameters,
        }
        for name, cfg in PRESET_CONFIGS.items()
    }
