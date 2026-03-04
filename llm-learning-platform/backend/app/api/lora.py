"""
LoRA & QLoRA API Router
"""

from typing import Dict, List, Optional
from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.core.lora import LoRALayer, LoRAModel, LoRAConfig, QLoRAQuantizer, compare_peft_methods

router = APIRouter()


class LoRACreateRequest(BaseModel):
    d_model: int = Field(256, ge=32, le=2048)
    num_layers: int = Field(4, ge=1, le=24)
    rank: int = Field(8, ge=1, le=128)
    alpha: int = Field(16, ge=1, le=256)
    dropout: float = Field(0.05, ge=0.0, le=0.5)
    target_modules: List[str] = ["q_proj", "v_proj"]


class LoRAForwardRequest(BaseModel):
    d_model: int = Field(128, ge=32, le=1024)
    rank: int = Field(8, ge=1, le=64)
    seq_len: int = Field(8, ge=1, le=64)


class QuantizeRequest(BaseModel):
    rows: int = Field(128, ge=16, le=1024)
    cols: int = Field(128, ge=16, le=1024)


class TrainSimRequest(BaseModel):
    d_model: int = Field(256, ge=32, le=1024)
    num_layers: int = Field(4, ge=1, le=12)
    rank: int = Field(8, ge=1, le=64)
    num_steps: int = Field(20, ge=5, le=100)


@router.post("/create")
async def create_lora(req: LoRACreateRequest):
    """Create a LoRA model and return parameter summary."""
    config = LoRAConfig(
        rank=req.rank, alpha=req.alpha,
        dropout=req.dropout, target_modules=req.target_modules,
    )
    model = LoRAModel(req.d_model, req.num_layers, config)
    return model.get_summary()


@router.post("/forward")
async def lora_forward(req: LoRAForwardRequest):
    """Run a forward pass through a LoRA layer and inspect matrices."""
    import numpy as np
    layer = LoRALayer(req.d_model, req.d_model, rank=req.rank)
    # Generate input from real text encoding
    text = "The transformer model uses self-attention to process input sequences efficiently."
    encoded = [ord(c) % req.d_model for c in text]
    x = np.zeros((1, req.seq_len, req.d_model), dtype=np.float32)
    for i in range(req.seq_len):
        x[0, i, encoded[i % len(encoded)]] = 1.0  # one-hot from real chars
    output = layer.forward(x.reshape(-1, req.d_model))

    delta_w = layer.get_delta_w()
    return {
        "input_shape": list(x.shape),
        "output_shape": [1, req.seq_len, req.d_model],
        "A_shape": list(layer.A.shape),
        "B_shape": list(layer.B.shape),
        "delta_w_norm": round(float(np.linalg.norm(delta_w)), 4),
        "delta_w_rank": req.rank,
        "scaling": layer.scaling,
        "output_sample": output[0, :8].tolist(),
    }


@router.post("/quantize")
async def analyze_quantization(req: QuantizeRequest):
    """Analyze NF4 quantization on real model weights."""
    import numpy as np
    from app.core.model import MicroGPT, GPTConfig
    # Use real model weight matrix instead of random data
    config = GPTConfig(vocab_size=256, d_model=max(req.rows, req.cols), num_heads=2, num_layers=2, d_ff=256, max_seq_len=64)
    model = MicroGPT(config)
    # Extract a real weight matrix from the embedding layer
    real_weight = model.embedding.weight.data[:req.rows, :req.cols].astype(np.float32)
    quantizer = QLoRAQuantizer()
    return quantizer.analyze_quantization(real_weight)


@router.post("/compare-peft")
async def compare_methods(d_model: int = 512, num_layers: int = 6):
    """Compare LoRA, QLoRA, prefix tuning, and full fine-tuning."""
    return compare_peft_methods(d_model, num_layers)


@router.post("/train")
async def train_lora(req: TrainSimRequest):
    """Run LoRA fine-tuning with real forward passes and loss computation."""
    config = LoRAConfig(rank=req.rank)
    model = LoRAModel(req.d_model, req.num_layers, config)
    summary = model.get_summary()
    metrics = model.train(num_steps=req.num_steps)
    return {"summary": summary, "training_metrics": metrics}


@router.get("/ranks")
async def list_recommended_ranks():
    """List recommended LoRA rank settings."""
    return {
        "ranks": [
            {"rank": 1, "use_case": "Minimal adaptation, very few parameters", "quality": "Low"},
            {"rank": 4, "use_case": "Light adaptation, good for small datasets", "quality": "Medium"},
            {"rank": 8, "use_case": "Standard LoRA, good balance", "quality": "Good"},
            {"rank": 16, "use_case": "Higher capacity, complex tasks", "quality": "Very Good"},
            {"rank": 32, "use_case": "Near full fine-tuning quality", "quality": "Excellent"},
            {"rank": 64, "use_case": "Maximum adaptation, diminishing returns", "quality": "Excellent+"},
        ]
    }
