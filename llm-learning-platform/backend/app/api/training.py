"""
Training API

Start, stop, monitor, and visualize model training sessions.
"""

from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.model import GPTConfig, PRESET_CONFIGS
from app.core.trainer import TrainingConfig, TrainingEngine, TrainingStatus

router = APIRouter()

_engine = TrainingEngine()


class StartTrainingRequest(BaseModel):
    model_preset: str = "micro"
    vocab_size: int = 256
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 4
    d_ff: int = 512
    max_seq_len: int = 64
    # Training params
    num_epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 50
    max_steps: int = 500
    grad_clip: float = 1.0


class TrainingStatusResponse(BaseModel):
    session_id: str
    status: str
    current_step: int
    total_steps: int
    best_loss: float
    metrics_count: int


@router.post("/start", response_model=TrainingStatusResponse)
async def start_training(request: StartTrainingRequest):
    """Start a new training session."""
    if request.model_preset in PRESET_CONFIGS:
        model_config = PRESET_CONFIGS[request.model_preset]
    else:
        model_config = GPTConfig(
            vocab_size=request.vocab_size,
            d_model=request.d_model,
            num_heads=request.num_heads,
            num_layers=request.num_layers,
            d_ff=request.d_ff,
            max_seq_len=request.max_seq_len,
        )

    training_config = TrainingConfig(
        num_epochs=request.num_epochs,
        batch_size=request.batch_size,
        learning_rate=request.learning_rate,
        weight_decay=request.weight_decay,
        warmup_steps=request.warmup_steps,
        max_steps=request.max_steps,
        grad_clip=request.grad_clip,
    )

    session = _engine.create_session(model_config, training_config)

    # Generate sample data and train
    data = TrainingEngine.generate_sample_data(
        vocab_size=model_config.vocab_size,
        num_samples=500,
        seq_len=model_config.max_seq_len,
    )

    _engine.train(session.session_id, data)

    return TrainingStatusResponse(
        session_id=session.session_id,
        status=session.status.value,
        current_step=session.current_step,
        total_steps=session.total_steps,
        best_loss=session.best_loss,
        metrics_count=len(session.metrics_history),
    )


@router.get("/{session_id}/status", response_model=TrainingStatusResponse)
async def get_training_status(session_id: str):
    """Get training session status."""
    session = _engine.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Training session not found")

    return TrainingStatusResponse(
        session_id=session.session_id,
        status=session.status.value,
        current_step=session.current_step,
        total_steps=session.total_steps,
        best_loss=session.best_loss,
        metrics_count=len(session.metrics_history),
    )


@router.get("/{session_id}/metrics")
async def get_training_metrics(
    session_id: str,
    last_n: Optional[int] = None,
):
    """Get training metrics history."""
    session = _engine.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Training session not found")

    history = session.metrics_history
    if last_n:
        history = history[-last_n:]

    return {
        "session_id": session_id,
        "metrics": [
            {
                "step": m.step,
                "epoch": m.epoch,
                "loss": m.loss,
                "learning_rate": m.learning_rate,
                "grad_norm": m.grad_norm,
                "tokens_per_sec": m.tokens_per_sec,
                "elapsed_seconds": m.elapsed_seconds,
                "perplexity": m.perplexity,
            }
            for m in history
        ],
    }


@router.post("/{session_id}/stop")
async def stop_training(session_id: str):
    """Stop a training session."""
    _engine.pause_session(session_id)
    return {"status": "stopped", "session_id": session_id}
