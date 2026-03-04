"""
Distributed Training API Router
"""

from typing import Optional
from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.core.distributed import (
    DistributedConfig, DataParallelAnalyzer, ModelParallelAnalyzer,
    PipelineParallelAnalyzer, ZeROAnalyzer, compare_strategies,
)

router = APIRouter()


class DistributedRequest(BaseModel):
    num_gpus: int = Field(4, ge=1, le=64)
    model_params_m: float = Field(125.0, ge=1, le=70000)
    batch_size: int = Field(32, ge=1, le=1024)
    gpu_memory_gb: float = Field(24.0, ge=4, le=80)


class ZeRORequest(BaseModel):
    num_gpus: int = Field(4, ge=1, le=64)
    model_params_m: float = Field(125.0, ge=1, le=70000)
    stage: int = Field(1, ge=0, le=3)
    gpu_memory_gb: float = Field(24.0, ge=4, le=80)


class PipelineRequest(BaseModel):
    num_gpus: int = Field(4, ge=1, le=64)
    model_params_m: float = Field(125.0, ge=1, le=70000)
    num_layers: int = Field(24, ge=4, le=96)
    num_micro_batches: int = Field(8, ge=1, le=64)
    gpu_memory_gb: float = Field(24.0, ge=4, le=80)


def _make_config(req) -> DistributedConfig:
    return DistributedConfig(
        num_gpus=req.num_gpus, model_params_m=req.model_params_m,
        batch_size=getattr(req, "batch_size", 32),
        gpu_memory_gb=req.gpu_memory_gb,
    )


@router.post("/data-parallel")
async def analyze_data_parallel(req: DistributedRequest):
    """Analyze data parallel configuration."""
    return DataParallelAnalyzer(_make_config(req)).analyze()


@router.post("/model-parallel")
async def analyze_model_parallel(req: DistributedRequest):
    """Analyze tensor model parallel configuration."""
    return ModelParallelAnalyzer(_make_config(req)).analyze()


@router.post("/pipeline-parallel")
async def analyze_pipeline_parallel(req: PipelineRequest):
    """Analyze pipeline parallel configuration."""
    config = DistributedConfig(
        num_gpus=req.num_gpus, model_params_m=req.model_params_m,
        gpu_memory_gb=req.gpu_memory_gb,
    )
    return PipelineParallelAnalyzer(config, req.num_layers).analyze(req.num_micro_batches)


@router.post("/zero")
async def analyze_zero(req: ZeRORequest):
    """Analyze ZeRO optimizer stage."""
    config = DistributedConfig(
        num_gpus=req.num_gpus, model_params_m=req.model_params_m,
        gpu_memory_gb=req.gpu_memory_gb,
    )
    return ZeROAnalyzer(config).analyze(req.stage)


@router.post("/zero/compare")
async def compare_zero_stages(req: DistributedRequest):
    """Compare all ZeRO stages."""
    config = _make_config(req)
    return ZeROAnalyzer(config).compare_stages()


@router.post("/compare-all")
async def compare_all_strategies(req: DistributedRequest):
    """Compare all distributed training strategies."""
    return compare_strategies(_make_config(req))


@router.get("/strategies")
async def list_strategies():
    """List available distributed training strategies."""
    return {
        "strategies": [
            {"name": "Data Parallelism", "id": "data_parallel", "description": "Replicate model, split data across GPUs"},
            {"name": "Tensor Parallelism", "id": "tensor_parallel", "description": "Split each layer's weights across GPUs"},
            {"name": "Pipeline Parallelism", "id": "pipeline_parallel", "description": "Assign different layers to different GPUs"},
            {"name": "ZeRO Stage 1", "id": "zero_1", "description": "Partition optimizer states across GPUs"},
            {"name": "ZeRO Stage 2", "id": "zero_2", "description": "Partition optimizer states + gradients"},
            {"name": "ZeRO Stage 3", "id": "zero_3", "description": "Partition everything (params + grads + optimizer)"},
        ]
    }
