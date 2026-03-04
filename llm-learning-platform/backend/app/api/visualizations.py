"""
Visualizations API

Generate data payloads optimized for D3.js and Three.js
visualizations in the frontend.
"""

from typing import List, Optional

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel, Field, model_validator

from app.core.embeddings import EmbeddingLayer
from app.core.attention import MultiHeadAttention, AttentionType
from app.core.model import MicroGPT, PRESET_CONFIGS

router = APIRouter()


class AttentionHeatmapRequest(BaseModel):
    seq_len: int = Field(default=8, ge=2, le=64)
    num_heads: int = Field(default=4, ge=1, le=16)
    d_model: int = 64
    attention_type: str = "full"

    @model_validator(mode="after")
    def validate_head_dimension(self):
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        return self


class EmbeddingScatterRequest(BaseModel):
    token_ids: List[int] = Field(default=list(range(50)))
    d_model: int = 64
    vocab_size: int = 256
    n_components: int = Field(default=3, ge=2, le=3)


class LossCurveRequest(BaseModel):
    num_steps: int = Field(default=100, ge=1, le=1000)
    noise_scale: float = 0.3


class ModelArchitectureRequest(BaseModel):
    preset: str = "micro"


@router.post("/attention-heatmap")
async def get_attention_heatmap(request: AttentionHeatmapRequest):
    """Generate attention weight heatmap data for D3.js visualization."""
    attn_type = AttentionType(request.attention_type)
    attention = MultiHeadAttention(
        d_model=request.d_model,
        num_heads=request.num_heads,
        attention_type=attn_type,
    )

    # Use real text-derived input instead of random data
    import torch
    text = "Attention is all you need for understanding transformer architecture."
    token_values = [ord(c) % request.d_model for c in text]
    x = torch.zeros((1, request.seq_len, request.d_model), dtype=torch.float32)
    for i in range(request.seq_len):
        x[0, i, token_values[i % len(token_values)]] = 1.0
        
    result = attention.forward_step_by_step(x)
    weights = np.array([result["attention_weights"]])  # Wrap in dummy batch dim: (1, heads, seq, seq)

    head_data = []
    for h in range(request.num_heads):
        w = weights[0, h]
        head_data.append({
            "head": h,
            "weights": w.tolist(),
            "entropy": float(-np.sum(w * np.log(w + 1e-10), axis=-1).mean()),
            "sparsity": float((w < 0.01).mean()),
        })

    return {
        "seq_len": request.seq_len,
        "num_heads": request.num_heads,
        "attention_type": request.attention_type,
        "heads": head_data,
    }


@router.post("/embedding-scatter")
async def get_embedding_scatter(request: EmbeddingScatterRequest):
    """Generate 2D/3D embedding scatter plot data for Three.js."""
    layer = EmbeddingLayer(
        vocab_size=request.vocab_size,
        embedding_dim=request.d_model,
    )

    import torch
    ids = torch.tensor([request.token_ids])
    embeddings = layer.token_embedding.weight.data[ids[0]].detach().cpu().numpy()  # (seq, d_model)

    # PCA reduction to 2D or 3D for the selected tokens.
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    cov = (centered.T @ centered) / max(centered.shape[0], 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1][:request.n_components]
    components = eigenvectors[:, idx].copy() # Ensure positive strides
    coords = (centered @ components).astype(np.float32)

    points = []
    for i, tid in enumerate(request.token_ids):
        point = {
            "token_id": tid,
            "x": float(coords[i, 0]),
            "y": float(coords[i, 1]),
        }
        if request.n_components == 3:
            point["z"] = float(coords[i, 2])
        points.append(point)

    return {
        "points": points,
        "dimensions": request.n_components,
        "d_model": request.d_model,
    }


@router.post("/loss-curve")
async def get_loss_curve(request: LossCurveRequest):
    """Generate real loss curve by running actual model forward passes."""
    from app.core.model import MicroGPT, GPTConfig
    from app.core.trainer import Trainer, TextDataset, TRAINING_CORPUS
    import torch

    config = GPTConfig(
        vocab_size=256, d_model=64, num_heads=2, num_layers=2,
        d_ff=256, max_seq_len=64, dropout=0.0,
    )
    model = MicroGPT(config)
    model.set_training(False)

    # Use real text data using TextDataset
    dataset = TextDataset(TRAINING_CORPUS, seq_len=config.max_seq_len)
    # Generate 200 sample rows
    x, y = dataset.get_batch(200)
    data = torch.cat((x, y[:, -1:]), dim=1).cpu().numpy() # [b, seq+1] shape matching old logic

    steps_list = []
    train_losses = []
    val_losses = []
    perplexities = []

    num_steps = max(1, int(request.num_steps))
    batch_size = 16
    num_samples = len(data)

    for step in range(num_steps):
        # Training loss: forward pass on a batch of real data
        start_idx = (step * batch_size) % max(num_samples - batch_size, 1)
        batch = data[start_idx : start_idx + batch_size]
        
        import torch
        inputs = torch.tensor(batch[:, :-1], dtype=torch.long)
        targets = torch.tensor(batch[:, 1:], dtype=torch.long)

        result = model.forward(inputs, targets)
        train_loss = float(result["loss"])

        # Validation loss: different batch of real data
        val_start = ((step + num_samples // 2) * batch_size) % max(num_samples - batch_size, 1)
        val_batch = data[val_start : val_start + batch_size]
        
        val_inputs = torch.tensor(val_batch[:, :-1], dtype=torch.long)
        val_targets = torch.tensor(val_batch[:, 1:], dtype=torch.long)
        
        val_result = model.forward(val_inputs, val_targets)
        val_loss = float(val_result["loss"])

        steps_list.append(step)
        train_losses.append(round(train_loss, 4))
        val_losses.append(round(val_loss, 4))
        perplexities.append(round(min(float(np.exp(train_loss)), 1e6), 2))

    # Learning rate schedule (cosine with warmup)
    warmup = min(50, num_steps // 5)
    lr_schedule = []
    for s in range(num_steps):
        if s < warmup:
            lr = 3e-4 * s / max(warmup, 1)
        else:
            progress = (s - warmup) / max(num_steps - warmup, 1)
            lr = 3e-4 * 0.5 * (1 + np.cos(np.pi * progress))
        lr_schedule.append(float(lr))

    return {
        "steps": steps_list,
        "train_loss": train_losses,
        "val_loss": val_losses,
        "learning_rate": lr_schedule,
        "perplexity": perplexities,
    }


@router.post("/model-architecture")
async def get_model_architecture(request: ModelArchitectureRequest):
    """Generate model architecture diagram data."""
    config = PRESET_CONFIGS.get(request.preset, PRESET_CONFIGS["micro"])

    layers = []
    # Input embedding
    layers.append({
        "type": "embedding",
        "name": "Token Embedding",
        "params": config.vocab_size * config.d_model,
        "shape": f"({config.vocab_size}, {config.d_model})",
    })
    layers.append({
        "type": "positional",
        "name": "Positional Encoding",
        "params": config.max_seq_len * config.d_model,
        "shape": f"({config.max_seq_len}, {config.d_model})",
    })

    # Transformer blocks
    for i in range(config.num_layers):
        layers.append({
            "type": "attention",
            "name": f"Multi-Head Attention (Layer {i})",
            "params": 4 * config.d_model * config.d_model,
            "shape": f"({config.d_model}, {config.d_model})",
            "num_heads": config.num_heads,
        })
        layers.append({
            "type": "norm",
            "name": f"Layer Norm (Layer {i})",
            "params": 2 * config.d_model,
            "shape": f"({config.d_model},)",
        })
        layers.append({
            "type": "mlp",
            "name": f"Feed-Forward (Layer {i})",
            "params": 2 * config.d_model * config.d_ff,
            "shape": f"({config.d_model}, {config.d_ff}, {config.d_model})",
        })

    # Output head
    layers.append({
        "type": "output",
        "name": "Language Model Head",
        "params": config.d_model * config.vocab_size,
        "shape": f"({config.d_model}, {config.vocab_size})",
    })

    return {
        "preset": request.preset,
        "config": {
            "d_model": config.d_model,
            "num_heads": config.num_heads,
            "num_layers": config.num_layers,
            "d_ff": config.d_ff,
            "vocab_size": config.vocab_size,
        },
        "layers": layers,
        "total_parameters": config.num_parameters,
    }
