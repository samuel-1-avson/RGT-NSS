
"""Distributed Training Analysis Engine

Computes real memory, throughput, and communication estimates for
data parallelism, model parallelism, pipeline parallelism,
and ZeRO optimization stages."""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class DistributedConfig:
    num_gpus: int = 4
    model_params_m: float = 125.0  # millions of parameters
    batch_size: int = 32
    seq_len: int = 512
    dtype_bytes: int = 4  # FP32
    gpu_memory_gb: float = 24.0


class DataParallelAnalyzer:
    """Compute real DDP (Distributed Data Parallelism) configuration analysis."""

    def __init__(self, config: DistributedConfig):
        self.config = config

    def analyze(self) -> Dict:
        """Analyze data parallel training."""
        c = self.config
        param_bytes = c.model_params_m * 1e6 * c.dtype_bytes
        grad_bytes = param_bytes
        optimizer_bytes = param_bytes * 2  # Adam: m + v

        per_gpu_memory_gb = (param_bytes + grad_bytes + optimizer_bytes) / 1e9
        micro_batch = c.batch_size // c.num_gpus

        # Communication: AllReduce gradients
        allreduce_bytes = param_bytes * 2 * (c.num_gpus - 1) / c.num_gpus  # Ring AllReduce

        return {
            "strategy": "data_parallel",
            "num_gpus": c.num_gpus,
            "per_gpu_memory_gb": round(per_gpu_memory_gb, 2),
            "fits_in_memory": per_gpu_memory_gb < c.gpu_memory_gb,
            "micro_batch_size": micro_batch,
            "effective_batch_size": c.batch_size,
            "communication_gb": round(allreduce_bytes / 1e9, 3),
            "communication_type": "AllReduce (Ring)",
            "redundancy": "Full model replicated on each GPU",
            "throughput_scaling": round(c.num_gpus * 0.9, 2),  # ~90% efficiency
        }


class ModelParallelAnalyzer:
    """Compute real Tensor Model Parallelism configuration analysis."""

    def __init__(self, config: DistributedConfig):
        self.config = config

    def analyze(self) -> Dict:
        c = self.config
        param_bytes = c.model_params_m * 1e6 * c.dtype_bytes
        per_gpu_params = param_bytes / c.num_gpus

        # Each GPU holds fraction of each layer
        per_gpu_memory_gb = per_gpu_params / 1e9 * 3  # params + grads + optimizer

        # Point-to-point communication for activations
        activation_bytes = c.batch_size * c.seq_len * 1024 * c.dtype_bytes  # d_model ≈ 1024

        return {
            "strategy": "tensor_parallel",
            "num_gpus": c.num_gpus,
            "per_gpu_memory_gb": round(per_gpu_memory_gb, 2),
            "fits_in_memory": per_gpu_memory_gb < c.gpu_memory_gb,
            "params_per_gpu_m": round(c.model_params_m / c.num_gpus, 1),
            "communication_gb": round(activation_bytes * 2 / 1e9, 3),
            "communication_type": "AllReduce activations per layer",
            "redundancy": "Model split across GPUs",
            "throughput_scaling": round(c.num_gpus * 0.75, 2),
        }


class PipelineParallelAnalyzer:
    """Compute real Pipeline Parallelism configuration analysis."""

    def __init__(self, config: DistributedConfig, num_layers: int = 24):
        self.config = config
        self.num_layers = num_layers

    def analyze(self, num_micro_batches: int = 8) -> Dict:
        c = self.config
        layers_per_gpu = self.num_layers // c.num_gpus
        param_bytes = c.model_params_m * 1e6 * c.dtype_bytes
        per_gpu_params = param_bytes / c.num_gpus
        per_gpu_memory_gb = per_gpu_params / 1e9 * 3

        # Pipeline bubble: (num_gpus - 1) / (num_gpus + num_micro_batches - 1)
        bubble_fraction = (c.num_gpus - 1) / (c.num_gpus + num_micro_batches - 1)
        efficiency = 1 - bubble_fraction

        # Schedule
        total_steps = c.num_gpus + num_micro_batches - 1
        schedule = []
        for step in range(total_steps):
            gpus_active = []
            for gpu in range(c.num_gpus):
                mb = step - gpu
                if 0 <= mb < num_micro_batches:
                    gpus_active.append({"gpu": gpu, "micro_batch": mb, "phase": "forward"})
            schedule.append({"step": step, "active": gpus_active})

        return {
            "strategy": "pipeline_parallel",
            "num_gpus": c.num_gpus,
            "layers_per_gpu": layers_per_gpu,
            "per_gpu_memory_gb": round(per_gpu_memory_gb, 2),
            "fits_in_memory": per_gpu_memory_gb < c.gpu_memory_gb,
            "num_micro_batches": num_micro_batches,
            "bubble_fraction": round(bubble_fraction, 3),
            "efficiency": round(efficiency, 3),
            "communication_type": "Point-to-point (activations)",
            "throughput_scaling": round(c.num_gpus * efficiency, 2),
            "schedule_preview": schedule[:min(10, len(schedule))],
        }


class ZeROAnalyzer:
    """Compute real ZeRO (Zero Redundancy Optimizer) stage analysis."""

    def __init__(self, config: DistributedConfig):
        self.config = config

    def analyze(self, stage: int = 1) -> Dict:
        c = self.config
        param_bytes = c.model_params_m * 1e6 * c.dtype_bytes
        grad_bytes = param_bytes
        opt_bytes = param_bytes * 2  # Adam m + v
        total_per_gpu_no_zero = (param_bytes + grad_bytes + opt_bytes) / 1e9

        if stage == 1:
            # Partition optimizer states
            per_gpu = (param_bytes + grad_bytes + opt_bytes / c.num_gpus) / 1e9
            partitioned = "Optimizer states"
        elif stage == 2:
            # Partition optimizer + gradients
            per_gpu = (param_bytes + (grad_bytes + opt_bytes) / c.num_gpus) / 1e9
            partitioned = "Optimizer states + Gradients"
        elif stage == 3:
            # Partition everything
            per_gpu = (param_bytes + grad_bytes + opt_bytes) / c.num_gpus / 1e9
            partitioned = "Parameters + Gradients + Optimizer states"
        else:
            per_gpu = total_per_gpu_no_zero
            partitioned = "Nothing"

        savings = 1 - per_gpu / total_per_gpu_no_zero if total_per_gpu_no_zero > 0 else 0

        return {
            "strategy": f"ZeRO Stage {stage}",
            "num_gpus": c.num_gpus,
            "no_zero_per_gpu_gb": round(total_per_gpu_no_zero, 2),
            "zero_per_gpu_gb": round(per_gpu, 2),
            "fits_in_memory": per_gpu < c.gpu_memory_gb,
            "memory_savings": f"{round(savings * 100, 1)}%",
            "partitioned": partitioned,
            "communication_overhead": ["AllGather params" if stage == 3 else "ReduceScatter grads"],
        }

    def compare_stages(self) -> Dict:
        return {
            "stages": {f"stage_{s}": self.analyze(s) for s in [0, 1, 2, 3]},
            "config": {
                "num_gpus": self.config.num_gpus,
                "model_params_m": self.config.model_params_m,
                "gpu_memory_gb": self.config.gpu_memory_gb,
            },
        }


def compare_strategies(config: Optional[DistributedConfig] = None) -> Dict:
    """Compare all distributed training strategies."""
    if config is None:
        config = DistributedConfig()
    return {
        "data_parallel": DataParallelAnalyzer(config).analyze(),
        "tensor_parallel": ModelParallelAnalyzer(config).analyze(),
        "pipeline_parallel": PipelineParallelAnalyzer(config).analyze(),
        "zero_stage_1": ZeROAnalyzer(config).analyze(1),
        "zero_stage_2": ZeROAnalyzer(config).analyze(2),
        "zero_stage_3": ZeROAnalyzer(config).analyze(3),
    }
