"""
Transformer Block Engine — PyTorch GPU-accelerated Transformer

Pre/post-norm transformer blocks with LayerNorm/RMSNorm,
SwiGLU/GELU MLP, and residual connections as real nn.Modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.core.attention import MultiHeadAttention, AttentionType


class NormType(Enum):
    LAYERNORM = "layernorm"
    RMSNORM = "rmsnorm"


class NormPlacement(Enum):
    PRE = "pre"
    POST = "post"


class ActivationType(Enum):
    RELU = "relu"
    GELU = "gelu"
    SWIGLU = "swiglu"
    SILU = "silu"


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class LayerNorm(nn.Module):
    """Standard Layer Normalization (wraps nn.LayerNorm)."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)

    def parameters(self, recurse=True):
        return self.norm.parameters(recurse)


class SwiGLUMLP(nn.Module):
    """SwiGLU activation MLP (used in LLaMA-style models)."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class StandardMLP(nn.Module):
    """Standard two-layer MLP with configurable activation."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        activation: ActivationType = ActivationType.GELU,
        dropout: float = 0.1,
        use_bias: bool = True,
    ):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=use_bias)
        self.fc2 = nn.Linear(d_ff, d_model, bias=use_bias)
        self.dropout = nn.Dropout(dropout)

        self.act_fn = {
            ActivationType.RELU: F.relu,
            ActivationType.GELU: F.gelu,
            ActivationType.SILU: F.silu,
        }.get(activation, F.gelu)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.act_fn(self.fc1(x))))


@dataclass
class TransformerBlockResult:
    output: torch.Tensor  # Changed from np.ndarray to torch.Tensor
    intermediates: Optional[Dict] = None


class TransformerBlock(nn.Module):
    """
    Complete transformer block with attention, MLP, norms, and residuals.

    Supports pre-norm and post-norm configurations.
    """

    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        d_ff: int = 512,
        norm_type: NormType = NormType.RMSNORM,
        norm_placement: NormPlacement = NormPlacement.PRE,
        activation: ActivationType = ActivationType.GELU,
        attention_type: AttentionType = AttentionType.FULL,
        dropout: float = 0.1,
        use_bias: bool = True,
        num_layers: int = 1,
    ):
        super().__init__()
        self.norm_placement = norm_placement

        # Normalization
        NormClass = RMSNorm if norm_type == NormType.RMSNORM else LayerNorm
        self.norm1 = NormClass(d_model)
        self.norm2 = NormClass(d_model)

        # Attention
        self.attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            attention_type=attention_type,
            use_bias=use_bias,
        )

        # MLP
        if activation == ActivationType.SWIGLU:
            self.mlp = SwiGLUMLP(d_model, d_ff, dropout)
        else:
            self.mlp = StandardMLP(d_model, d_ff, activation, dropout, use_bias)

        self.dropout = nn.Dropout(dropout)
        self._training_mode = True

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        store_intermediates: bool = False,
    ) -> TransformerBlockResult:
        """Forward pass through the transformer block."""
        intermediates = {}

        if self.norm_placement == NormPlacement.PRE:
            # Pre-norm: Norm → Attention → Residual → Norm → MLP → Residual
            normed = self.norm1(x)
            attn_out = self.attention(normed, mask=mask, store_weights=store_intermediates)
            x = x + attn_out

            if store_intermediates:
                intermediates["post_attention"] = x.detach().cpu().numpy()
                weights = self.attention.get_attention_weights()
                if weights is not None:
                    intermediates["attention_weights"] = weights

            normed = self.norm2(x)
            mlp_out = self.mlp(normed)
            x = x + mlp_out

            if store_intermediates:
                intermediates["post_mlp"] = x.detach().cpu().numpy()
        else:
            # Post-norm: Attention → Residual → Norm → MLP → Residual → Norm
            attn_out = self.attention(x, mask=mask, store_weights=store_intermediates)
            x = self.norm1(x + attn_out)

            if store_intermediates:
                intermediates["post_attention"] = x.detach().cpu().numpy()

            mlp_out = self.mlp(x)
            x = self.norm2(x + mlp_out)

            if store_intermediates:
                intermediates["post_mlp"] = x.detach().cpu().numpy()

        return TransformerBlockResult(
            output=x,
            intermediates=intermediates if store_intermediates else None,
        )

    def set_training(self, training: bool):
        self._training_mode = training
        self.train(training)
