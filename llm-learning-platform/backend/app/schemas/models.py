"""
Pydantic schemas for model-related requests and responses.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field, validator


class GPTConfigRequest(BaseModel):
    """Configuration for creating a GPT model."""
    vocab_size: int = Field(default=256, ge=16, le=100000, 
                           description="Vocabulary size")
    max_seq_len: int = Field(default=256, ge=32, le=8192,
                            description="Maximum sequence length")
    d_model: int = Field(default=128, ge=32, le=4096,
                        description="Model dimension")
    num_layers: int = Field(default=4, ge=1, le=48,
                           description="Number of transformer layers")
    num_heads: int = Field(default=4, ge=1, le=64,
                          description="Number of attention heads")
    d_ff: Optional[int] = Field(default=None, ge=64, le=16384,
                               description="Feed-forward dimension")
    dropout: float = Field(default=0.1, ge=0.0, le=0.9,
                          description="Dropout rate")
    attention_dropout: float = Field(default=0.1, ge=0.0, le=0.9,
                                    description="Attention dropout rate")
    activation: str = Field(default='gelu', 
                           pattern='^(gelu|relu|swiglu)$',
                           description="Activation function")
    norm_type: str = Field(default='rmsnorm',
                          pattern='^(rmsnorm|layernorm)$',
                          description="Normalization type")
    tie_weights: bool = Field(default=True,
                             description="Tie input/output embeddings")
    backend: str = Field(default='pytorch',
                        pattern='^(custom|pytorch)$',
                        description="Backend: 'custom' (NumPy) or 'pytorch' (GPU)")
    
    @validator('d_model')
    def d_model_divisible_by_heads(cls, v, values):
        """Ensure d_model is divisible by num_heads."""
        if 'num_heads' in values and values['num_heads'] > 0:
            if v % values['num_heads'] != 0:
                raise ValueError(f'd_model ({v}) must be divisible by num_heads ({values["num_heads"]})')
        return v
    
    @validator('d_ff')
    def set_default_d_ff(cls, v, values):
        """Set default d_ff if not provided."""
        if v is None and 'd_model' in values:
            return values['d_model'] * 4
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "vocab_size": 256,
                "max_seq_len": 256,
                "d_model": 128,
                "num_layers": 4,
                "num_heads": 4,
                "dropout": 0.1,
                "backend": "pytorch"
            }
        }


class ModelCreateRequest(BaseModel):
    """Request to create a new model."""
    config: GPTConfigRequest
    name: Optional[str] = Field(default=None, max_length=100,
                               description="Optional model name")
    description: Optional[str] = Field(default=None, max_length=500,
                                      description="Optional model description")
    tags: Optional[List[str]] = Field(default=None,
                                     description="Optional tags for organization")
    
    class Config:
        json_schema_extra = {
            "example": {
                "config": {
                    "vocab_size": 256,
                    "max_seq_len": 256,
                    "d_model": 128,
                    "num_layers": 4,
                    "num_heads": 4,
                    "backend": "pytorch"
                },
                "name": "My GPT Model",
                "description": "A small GPT for testing",
                "tags": ["test", "small"]
            }
        }


class ModelInfo(BaseModel):
    """Model information."""
    model_id: str
    name: Optional[str] = None
    description: Optional[str] = None
    config: Dict[str, Any]
    num_parameters: int
    created_at: str
    updated_at: Optional[str] = None
    status: str
    backend: str
    device: Optional[str] = None
    tags: Optional[List[str]] = None
    owner_id: Optional[str] = None
    
    class Config:
        from_attributes = True


class ModelResponse(BaseModel):
    """Response for model operations."""
    success: bool = True
    model_id: str
    message: Optional[str] = None
    model: Optional[ModelInfo] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "model_id": "abc12345",
                "message": "Model created successfully",
                "model": {
                    "model_id": "abc12345",
                    "config": {"vocab_size": 256, "d_model": 128},
                    "num_parameters": 1000000,
                    "created_at": "2024-01-01T00:00:00",
                    "status": "created",
                    "backend": "pytorch"
                }
            }
        }


class ModelListResponse(BaseModel):
    """Response for listing models."""
    success: bool = True
    models: List[ModelInfo]
    total: int
    page: int = 1
    per_page: int = 20
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "models": [],
                "total": 0,
                "page": 1,
                "per_page": 20
            }
        }


class ModelResetRequest(BaseModel):
    """Request to reset a model."""
    preserve_config: bool = Field(default=True,
                                 description="Keep the same configuration")
    seed: Optional[int] = Field(default=None,
                               description="Random seed for reproducibility")


class ModelDeleteResponse(BaseModel):
    """Response for model deletion."""
    success: bool
    model_id: str
    message: str
