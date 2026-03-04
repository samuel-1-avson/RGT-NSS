"""
Long Context Techniques API Router
"""

from typing import List, Optional
from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.core.long_context import (
    RoPEAnalyzer, RoPEConfig, ALiBiAnalyzer, compare_position_methods,
)

router = APIRouter()


class RoPERequest(BaseModel):
    dim: int = Field(128, ge=32, le=512)
    scaling_factor: float = Field(4.0, ge=1.0, le=32.0)
    max_position: int = Field(2048, ge=128, le=16384)
    method: str = "none"  # none, linear, ntk, yarn


class ALiBiRequest(BaseModel):
    num_heads: int = Field(8, ge=1, le=32)
    seq_len: int = Field(16, ge=4, le=64)


class ContextExtensionRequest(BaseModel):
    num_heads: int = Field(8, ge=1, le=32)
    train_length: int = Field(2048, ge=128, le=8192)
    test_lengths: Optional[List[int]] = None


@router.post("/rope/frequencies")
async def rope_frequencies(req: RoPERequest):
    """Compute RoPE frequencies with optional scaling."""
    config = RoPEConfig(dim=req.dim, max_position=req.max_position, scaling_factor=req.scaling_factor)
    analyzer = RoPEAnalyzer(config)
    return analyzer.compute_frequencies(req.method)


@router.post("/rope/compare")
async def rope_compare_scaling(req: RoPERequest):
    """Compare different RoPE scaling methods."""
    config = RoPEConfig(dim=req.dim, max_position=req.max_position, scaling_factor=req.scaling_factor)
    return RoPEAnalyzer(config).compare_scaling_methods()


@router.post("/alibi/bias-matrix")
async def alibi_bias_matrix(req: ALiBiRequest):
    """Compute ALiBi bias matrix for visualization."""
    return ALiBiAnalyzer(req.num_heads).compute_bias_matrix(req.seq_len)


@router.post("/alibi/extrapolation")
async def alibi_extrapolation(req: ContextExtensionRequest):
    """Analyze ALiBi context extension capability."""
    return ALiBiAnalyzer(req.num_heads).analyze_context_extension(req.train_length, req.test_lengths)


@router.post("/compare")
async def compare_all_methods(dim: int = 128, num_heads: int = 8, seq_len: int = 2048):
    """Compare all positional encoding methods."""
    return compare_position_methods(seq_len, dim, num_heads)
