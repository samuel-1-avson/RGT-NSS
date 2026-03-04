"""
Device Management — GPU Detection and Device Placement

Provides centralized device management for the LLM Learning Platform.
Auto-detects CUDA GPUs and provides graceful CPU fallback.
"""

import torch
from typing import Dict, Optional

# ─── Singleton Device ────────────────────────────────────────

_device: Optional[torch.device] = None


def get_device() -> torch.device:
    """Get the best available compute device (CUDA GPU or CPU)."""
    global _device
    if _device is None:
        if torch.cuda.is_available():
            _device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            _device = torch.device("mps")  # Apple Silicon
        else:
            _device = torch.device("cpu")
    return _device


def get_device_info() -> Dict:
    """Get detailed information about the compute device."""
    device = get_device()
    info: Dict = {
        "device": str(device),
        "type": device.type,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        info.update({
            "cuda_version": torch.version.cuda or "N/A",
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_count": torch.cuda.device_count(),
            "gpu_memory_total_mb": round(torch.cuda.get_device_properties(0).total_mem / 1e6, 1),
            "gpu_memory_allocated_mb": round(torch.cuda.memory_allocated(0) / 1e6, 1),
            "gpu_memory_reserved_mb": round(torch.cuda.memory_reserved(0) / 1e6, 1),
        })
    elif device.type == "mps":
        info["backend"] = "Apple Metal Performance Shaders"

    return info


def to_device(tensor: torch.Tensor) -> torch.Tensor:
    """Move a tensor to the active compute device."""
    return tensor.to(get_device())


def empty_cache():
    """Free unused GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
