"""
Training API — PyTorch GPU-accelerated Training

Start, monitor, and control model training sessions with real
PyTorch optimization on GPU.
"""

from typing import Optional
import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.model import GPTConfig, MicroGPT, PRESET_CONFIGS
from app.core.trainer import Trainer, TrainingConfig

router = APIRouter()

# In-memory training sessions
_sessions: dict = {}


class StartTrainingRequest(BaseModel):
    model_preset: str = "nano"
    vocab_size: int = 256
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 4
    d_ff: int = 512
    max_seq_len: int = 64
    # Training params
    batch_size: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_steps: int = 50
    grad_clip: float = 1.0
    optimizer: str = "adamw"
    scheduler: str = "cosine"


class TrainingStatusResponse(BaseModel):
    session_id: str
    status: str
    current_step: int
    total_steps: int
    best_loss: float
    metrics_count: int


@router.post("/start", response_model=TrainingStatusResponse)
async def start_training(request: StartTrainingRequest):
    """Start a new training session with real PyTorch training."""
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

    model = MicroGPT(model_config)
    train_config = TrainingConfig(
        learning_rate=request.learning_rate,
        weight_decay=request.weight_decay,
        batch_size=request.batch_size,
        max_steps=request.max_steps,
        grad_clip=request.grad_clip,
        optimizer=request.optimizer,
        scheduler=request.scheduler,
    )

    trainer = Trainer(model=model, config=train_config)

    # Run training
    results = trainer.train(num_steps=request.max_steps)
    session_id = str(uuid.uuid4())

    _sessions[session_id] = {
        "trainer": trainer,
        "results": results,
        "status": "completed",
    }

    summary = trainer.get_training_summary()

    return TrainingStatusResponse(
        session_id=session_id,
        status="completed",
        current_step=summary["total_steps"],
        total_steps=request.max_steps,
        best_loss=round(summary["best_loss"], 6),
        metrics_count=len(results),
    )


@router.get("/{session_id}/status", response_model=TrainingStatusResponse)
async def get_training_status(session_id: str):
    """Get training session status."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Training session not found")

    session = _sessions[session_id]
    summary = session["trainer"].get_training_summary()

    return TrainingStatusResponse(
        session_id=session_id,
        status=session["status"],
        current_step=summary["total_steps"],
        total_steps=summary["total_steps"],
        best_loss=round(summary["best_loss"], 6),
        metrics_count=len(session["results"]),
    )


@router.get("/{session_id}/metrics")
async def get_training_metrics(session_id: str, last_n: Optional[int] = None):
    """Get training metrics history."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Training session not found")

    results = _sessions[session_id]["results"]
    if last_n:
        results = results[-last_n:]

    return {"session_id": session_id, "metrics": results}


@router.post("/{session_id}/stop")
async def stop_training(session_id: str):
    """Stop a training session."""
    if session_id in _sessions:
        _sessions[session_id]["status"] = "stopped"
    return {"status": "stopped", "session_id": session_id}
