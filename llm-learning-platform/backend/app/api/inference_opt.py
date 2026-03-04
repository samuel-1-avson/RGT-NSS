"""
Inference Optimization API Router
"""

from typing import Dict, List, Optional
from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.core.inference_opt import (
    KVCacheAnalyzer, KVCacheConfig,
    QuantizationAnalyzer,
    SpeculativeDecodingEngine,
)

router = APIRouter()


class KVCacheSimRequest(BaseModel):
    num_layers: int = Field(6, ge=1, le=32)
    num_heads: int = Field(8, ge=1, le=64)
    head_dim: int = Field(64, ge=16, le=256)
    prompt_len: int = Field(20, ge=1, le=512)
    gen_len: int = Field(50, ge=1, le=500)


class QuantCompareRequest(BaseModel):
    rows: int = Field(512, ge=16, le=2048)
    cols: int = Field(512, ge=16, le=2048)


class SpecDecodeRequest(BaseModel):
    total_tokens: int = Field(100, ge=10, le=500)
    gamma: int = Field(4, ge=1, le=8)
    acceptance_rate: float = Field(0.7, ge=0.1, le=1.0)


@router.post("/kv-cache/analyze")
async def analyze_kv_cache(req: KVCacheSimRequest):
    """Analyze KV cache growth during generation."""
    config = KVCacheConfig(
        num_layers=req.num_layers, num_heads=req.num_heads,
        head_dim=req.head_dim,
    )
    sim = KVCacheAnalyzer(config)
    steps = sim.analyze_generation(req.prompt_len, req.gen_len)
    return {
        "config": {"num_layers": req.num_layers, "num_heads": req.num_heads, "head_dim": req.head_dim},
        "steps": steps,
        "final_cache_mb": steps[-1]["cache_mb"] if steps else 0,
    }


@router.post("/quantization/compare")
async def compare_quantizations(req: QuantCompareRequest):
    """Compare FP32, FP16, INT8, INT4 quantization quality."""
    return QuantizationAnalyzer.compare_quantizations((req.rows, req.cols))


@router.post("/speculative-decoding/run")
async def run_speculative_decoding(req: SpecDecodeRequest):
    """Run speculative decoding performance analysis."""
    sim = SpeculativeDecodingEngine(gamma=req.gamma, acceptance_rate=req.acceptance_rate)
    return sim.run(req.total_tokens)


@router.get("/techniques")
async def list_optimization_techniques():
    """List available inference optimization techniques."""
    return {
        "techniques": [
            {
                "name": "KV Cache",
                "description": "Cache key-value pairs to avoid recomputation in autoregressive generation",
                "speedup": "O(n) → O(1) per token",
                "memory_cost": "Linear in sequence length",
            },
            {
                "name": "Quantization",
                "description": "Reduce precision of weights (FP32→INT8/INT4) for smaller memory and faster inference",
                "speedup": "2-4x faster, 2-8x less memory",
                "memory_cost": "Slight quality loss",
            },
            {
                "name": "Speculative Decoding",
                "description": "Use small draft model to propose tokens, verify with large model in parallel",
                "speedup": "2-3x faster generation",
                "memory_cost": "Need both draft and target models",
            },
            {
                "name": "Continuous Batching",
                "description": "Dynamically add/remove sequences from running batch",
                "speedup": "Higher throughput (2-10x)",
                "memory_cost": "More complex scheduling",
            },
            {
                "name": "Flash Attention",
                "description": "IO-aware attention algorithm reducing memory from O(n²) to O(n)",
                "speedup": "2-4x faster attention",
                "memory_cost": "No quality loss",
            },
        ]
    }
