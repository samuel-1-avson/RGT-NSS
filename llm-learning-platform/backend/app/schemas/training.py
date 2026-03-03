"""
Pydantic schemas for training-related requests and responses.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator


class TrainingConfigRequest(BaseModel):
    """Configuration for training."""
    model_id: str = Field(..., min_length=1, description="Model ID to train")
    batch_size: int = Field(default=32, ge=1, le=512,
                           description="Training batch size")
    learning_rate: float = Field(default=3e-4, ge=1e-7, le=1.0,
                                description="Initial learning rate")
    min_learning_rate: float = Field(default=3e-5, ge=1e-7, le=1.0,
                                    description="Minimum learning rate")
    warmup_steps: int = Field(default=100, ge=0, le=10000,
                             description="Number of warmup steps")
    max_steps: int = Field(default=10000, ge=100, le=1000000,
                          description="Maximum training steps")
    max_epochs: Optional[int] = Field(default=None, ge=1, le=1000,
                                     description="Maximum epochs (optional)")
    grad_clip: float = Field(default=1.0, ge=0.0, le=10.0,
                            description="Gradient clipping threshold")
    weight_decay: float = Field(default=0.1, ge=0.0, le=1.0,
                               description="Weight decay (L2 regularization)")
    seq_length: int = Field(default=256, ge=32, le=4096,
                           description="Sequence length for training")
    eval_interval: int = Field(default=100, ge=10, le=10000,
                              description="Evaluation interval (steps)")
    eval_steps: int = Field(default=10, ge=1, le=1000,
                           description="Number of steps for evaluation")
    checkpoint_interval: int = Field(default=1000, ge=100, le=50000,
                                    description="Checkpoint save interval")
    dataset_id: Optional[str] = Field(default=None,
                                     description="Dataset ID to use for training")
    validation_split: float = Field(default=0.1, ge=0.0, le=0.5,
                                   description="Fraction of data for validation")
    
    @validator('max_steps')
    def validate_max_steps(cls, v, values):
        """Ensure max_steps is reasonable."""
        if v < values.get('warmup_steps', 0):
            raise ValueError('max_steps must be greater than warmup_steps')
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_id": "abc12345",
                "batch_size": 32,
                "learning_rate": 0.0003,
                "max_steps": 10000,
                "seq_length": 256
            }
        }


class TrainingStartRequest(BaseModel):
    """Request to start training."""
    config: TrainingConfigRequest
    resume_from_checkpoint: Optional[str] = Field(default=None,
                                                description="Checkpoint to resume from")
    dry_run: bool = Field(default=False,
                         description="Validate without starting training")
    
    class Config:
        json_schema_extra = {
            "example": {
                "config": {
                    "model_id": "abc12345",
                    "batch_size": 32,
                    "learning_rate": 0.0003,
                    "max_steps": 10000
                },
                "resume_from_checkpoint": None,
                "dry_run": False
            }
        }


class TrainingStatusResponse(BaseModel):
    """Training status response."""
    session_id: str
    model_id: str
    is_training: bool
    current_step: int
    current_epoch: int
    max_steps: int
    progress: float = Field(..., ge=0.0, le=1.0)
    best_loss: float
    current_loss: Optional[float] = None
    learning_rate: Optional[float] = None
    status: str = Field(..., pattern='^(initializing|running|paused|completed|failed|stopped)$')
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    device: Optional[str] = None
    backend: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "train_123",
                "model_id": "abc12345",
                "is_training": True,
                "current_step": 500,
                "current_epoch": 2,
                "max_steps": 10000,
                "progress": 0.05,
                "best_loss": 2.5,
                "status": "running"
            }
        }


class TrainingMetricsResponse(BaseModel):
    """Training metrics at a specific step."""
    step: int
    epoch: int
    loss: float
    perplexity: float
    learning_rate: float
    grad_norm: float
    tokens_per_sec: float
    time_elapsed: float
    time_remaining: Optional[float] = None
    gpu_memory_mb: Optional[float] = None
    gpu_utilization: Optional[float] = None
    batch_losses: Optional[List[float]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "step": 100,
                "epoch": 1,
                "loss": 2.3,
                "perplexity": 9.97,
                "learning_rate": 0.0003,
                "grad_norm": 1.2,
                "tokens_per_sec": 1250.5
            }
        }


class TrainingHistoryResponse(BaseModel):
    """Training history response."""
    session_id: str
    model_id: str
    metrics: List[TrainingMetricsResponse]
    total_steps: int
    summary: Optional[Dict[str, Any]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "train_123",
                "model_id": "abc12345",
                "metrics": [],
                "total_steps": 100
            }
        }


class TrainingStopRequest(BaseModel):
    """Request to stop training."""
    save_checkpoint: bool = Field(default=True,
                                 description="Save checkpoint before stopping")
    checkpoint_name: Optional[str] = Field(default=None,
                                          description="Custom checkpoint name")


class TrainingStopResponse(BaseModel):
    """Response for stopping training."""
    success: bool
    session_id: str
    message: str
    checkpoint_path: Optional[str] = None
    final_step: Optional[int] = None
    final_loss: Optional[float] = None
