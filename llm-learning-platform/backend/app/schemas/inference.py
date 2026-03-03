"""
Pydantic schemas for inference-related requests and responses.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator


class GenerationRequest(BaseModel):
    """Request for text generation."""
    model_id: str = Field(..., min_length=1, description="Model ID to use")
    prompt: str = Field(..., min_length=1, max_length=10000,
                       description="Input prompt text")
    max_new_tokens: int = Field(default=100, ge=1, le=2000,
                               description="Maximum tokens to generate")
    temperature: float = Field(default=0.8, ge=0.01, le=2.0,
                              description="Sampling temperature")
    top_k: Optional[int] = Field(default=40, ge=1, le=1000,
                                description="Top-K sampling")
    top_p: Optional[float] = Field(default=0.9, ge=0.0, le=1.0,
                                  description="Nucleus (top-P) sampling")
    repetition_penalty: float = Field(default=1.0, ge=1.0, le=2.0,
                                     description="Repetition penalty")
    seed: Optional[int] = Field(default=None,
                               description="Random seed for reproducibility")
    stop_sequences: Optional[List[str]] = Field(default=None,
                                               description="Sequences to stop generation")
    
    @validator('prompt')
    def validate_prompt(cls, v, values):
        """Validate prompt length."""
        if not v.strip():
            raise ValueError('Prompt cannot be empty or whitespace only')
        return v.strip()
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_id": "abc12345",
                "prompt": "Once upon a time",
                "max_new_tokens": 50,
                "temperature": 0.8,
                "top_k": 40
            }
        }


class TokenProbability(BaseModel):
    """Token with probability."""
    token_id: int
    token: str
    probability: float
    logit: Optional[float] = None


class GenerationResponse(BaseModel):
    """Response for text generation."""
    success: bool = True
    model_id: str
    prompt: str
    generated_text: str
    full_text: str
    tokens_generated: int
    tokens: Optional[List[str]] = None
    token_ids: Optional[List[int]] = None
    top_predictions: Optional[List[List[TokenProbability]]] = None
    generation_time_ms: Optional[float] = None
    tokens_per_second: Optional[float] = None
    backend: str
    warnings: Optional[List[str]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "model_id": "abc12345",
                "prompt": "Once upon a time",
                "generated_text": ", there was a brave knight",
                "full_text": "Once upon a time, there was a brave knight",
                "tokens_generated": 7,
                "backend": "pytorch"
            }
        }


class TokenizeRequest(BaseModel):
    """Request for text tokenization."""
    text: str = Field(..., min_length=1, max_length=100000,
                     description="Text to tokenize")
    strategy: str = Field(default='character',
                         pattern='^(character|word|bpe|gpt2|tiktoken)$',
                         description="Tokenization strategy")
    encoding: Optional[str] = Field(default=None,
                                   description="Encoding name (for tiktoken)")
    
    @validator('text')
    def validate_text(cls, v):
        """Validate text."""
        if not v.strip():
            raise ValueError('Text cannot be empty')
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Hello, world!",
                "strategy": "character"
            }
        }


class TokenizeResponse(BaseModel):
    """Response for tokenization."""
    success: bool = True
    strategy: str
    text: str
    tokens: List[str]
    token_ids: List[int]
    num_tokens: int
    vocabulary: Optional[List[str]] = None
    vocab_size: Optional[int] = None
    merge_history: Optional[List[Dict[str, Any]]] = None
    compression_ratio: Optional[float] = None
    encoding: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "strategy": "character",
                "text": "Hello",
                "tokens": ["H", "e", "l", "l", "o"],
                "token_ids": [72, 101, 108, 108, 111],
                "num_tokens": 5
            }
        }


class ForwardPassRequest(BaseModel):
    """Request for model forward pass."""
    model_id: str = Field(..., description="Model ID to use")
    input_text: str = Field(..., min_length=1, max_length=10000,
                           description="Input text")
    return_attention: bool = Field(default=False,
                                  description="Return attention weights")
    return_logits: bool = Field(default=True,
                               description="Return output logits")
    layer_indices: Optional[List[int]] = Field(default=None,
                                              description="Specific layers to return attention for")
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_id": "abc12345",
                "input_text": "Hello world",
                "return_attention": True
            }
        }


class ForwardPassResponse(BaseModel):
    """Response for forward pass."""
    success: bool = True
    model_id: str
    input: str
    logits_shape: Optional[List[int]] = None
    loss: Optional[float] = None
    perplexity: Optional[float] = None
    top_predictions: Optional[List[Dict[str, Any]]] = None
    attention_weights: Optional[List[List[List[float]]]] = None
    attention_shapes: Optional[List[List[int]]] = None
    layer_names: Optional[List[str]] = None
    processing_time_ms: Optional[float] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "model_id": "abc12345",
                "input": "Hello",
                "logits_shape": [1, 5, 256],
                "top_predictions": [
                    {"token_id": 32, "probability": 0.8}
                ]
            }
        }


class EmbeddingRequest(BaseModel):
    """Request for getting embeddings."""
    model_id: str = Field(..., description="Model ID")
    text: Optional[str] = Field(default=None, description="Text to embed")
    token_ids: Optional[List[int]] = Field(default=None, description="Token IDs to embed")
    projection_method: str = Field(default='pca',
                                  pattern='^(pca|tsne|umap|none)$',
                                  description="Dimensionality reduction method")
    dimensions: int = Field(default=2, ge=2, le=3,
                           description="Projection dimensions (2 or 3)")
    
    @validator('dimensions')
    def validate_dimensions(cls, v):
        """Validate dimensions."""
        if v not in [2, 3]:
            raise ValueError('Dimensions must be 2 or 3')
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_id": "abc12345",
                "text": "Hello world",
                "projection_method": "pca",
                "dimensions": 2
            }
        }


class EmbeddingResponse(BaseModel):
    """Response for embeddings."""
    success: bool = True
    model_id: str
    vocab_size: int
    embedding_dim: int
    projections: List[List[float]]
    tokens: List[str]
    method: str
    variance_explained: Optional[List[float]] = None
