"""
Transformer Block System

Complete transformer block with all normalization variants
(LayerNorm, RMSNorm, DeepNorm), activation functions (GELU,
SwiGLU, SiLU), and pre-norm / post-norm placement.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import numpy as np

from app.core.tensor import Tensor
from app.core.attention import MultiHeadAttention, AttentionType


class NormType(str, Enum):
    LAYERNORM = "layernorm"
    RMSNORM = "rmsnorm"
    DEEPNORM = "deepnorm"


class NormPlacement(str, Enum):
    PRE = "pre"
    POST = "post"


class ActivationType(str, Enum):
    GELU = "gelu"
    RELU = "relu"
    SWIGLU = "swiglu"
    SILU = "silu"


@dataclass
class TransformerBlockOutput:
    output: np.ndarray
    intermediates: Optional[Dict[str, np.ndarray]] = None


# ─── Normalization Layers ────────────────────────────────────

class LayerNorm:
    """Standard Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        self.eps = eps
        self.gamma = Tensor(np.ones(dim, dtype=np.float32), requires_grad=True)
        self.beta = Tensor(np.zeros(dim, dtype=np.float32), requires_grad=True)

    def forward(self, x: np.ndarray) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        normed = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma.data * normed + self.beta.data

    def parameters(self) -> List[Tensor]:
        return [self.gamma, self.beta]


class RMSNorm:
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        self.eps = eps
        self.gamma = Tensor(np.ones(dim, dtype=np.float32), requires_grad=True)

    def forward(self, x: np.ndarray) -> np.ndarray:
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        return self.gamma.data * (x / rms)

    def parameters(self) -> List[Tensor]:
        return [self.gamma]


class DeepNorm:
    """DeepNorm for training very deep transformers (Wang et al., 2022)."""

    def __init__(self, dim: int, num_layers: int, eps: float = 1e-6):
        self.eps = eps
        self.gamma = Tensor(np.ones(dim, dtype=np.float32), requires_grad=True)
        self.alpha = (2.0 * num_layers) ** 0.25

    def forward(self, x: np.ndarray) -> np.ndarray:
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        return self.alpha * self.gamma.data * (x / rms)

    def parameters(self) -> List[Tensor]:
        return [self.gamma]


# ─── MLP / Feedforward ──────────────────────────────────────

class MLP:
    """Feedforward network with multiple activation function options."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        activation: ActivationType = ActivationType.GELU,
        use_bias: bool = True,
    ):
        self.d_model = d_model
        self.d_ff = d_ff
        self.activation = activation
        self.use_bias = use_bias

        scale_up = np.sqrt(2.0 / d_model).astype(np.float32)
        scale_down = np.sqrt(2.0 / d_ff).astype(np.float32)

        if activation == ActivationType.SWIGLU:
            self.W_up = Tensor(np.random.randn(d_model, d_ff).astype(np.float32) * scale_up, requires_grad=True)
            self.W_gate = Tensor(np.random.randn(d_model, d_ff).astype(np.float32) * scale_up, requires_grad=True)
            self.W_down = Tensor(np.random.randn(d_ff, d_model).astype(np.float32) * scale_down, requires_grad=True)
            if use_bias:
                self.b_up = Tensor(np.zeros(d_ff, dtype=np.float32), requires_grad=True)
                self.b_gate = Tensor(np.zeros(d_ff, dtype=np.float32), requires_grad=True)
                self.b_down = Tensor(np.zeros(d_model, dtype=np.float32), requires_grad=True)
        else:
            self.W_1 = Tensor(np.random.randn(d_model, d_ff).astype(np.float32) * scale_up, requires_grad=True)
            self.W_2 = Tensor(np.random.randn(d_ff, d_model).astype(np.float32) * scale_down, requires_grad=True)
            if use_bias:
                self.b_1 = Tensor(np.zeros(d_ff, dtype=np.float32), requires_grad=True)
                self.b_2 = Tensor(np.zeros(d_model, dtype=np.float32), requires_grad=True)

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.activation == ActivationType.SWIGLU:
            gate = x @ self.W_gate.data
            up = x @ self.W_up.data
            if self.use_bias:
                gate = gate + self.b_gate.data
                up = up + self.b_up.data
            # SwiGLU: Swish(gate) * up
            gate = gate * (1.0 / (1.0 + np.exp(-gate)))  # swish
            hidden = gate * up
            out = hidden @ self.W_down.data
            if self.use_bias:
                out = out + self.b_down.data
            return out

        # Standard two-layer MLP
        hidden = x @ self.W_1.data
        if self.use_bias:
            hidden = hidden + self.b_1.data

        if self.activation == ActivationType.GELU:
            c = np.sqrt(2.0 / np.pi)
            hidden = 0.5 * hidden * (1.0 + np.tanh(c * (hidden + 0.044715 * hidden ** 3)))
        elif self.activation == ActivationType.RELU:
            hidden = np.maximum(0, hidden)
        elif self.activation == ActivationType.SILU:
            hidden = hidden * (1.0 / (1.0 + np.exp(-hidden)))

        out = hidden @ self.W_2.data
        if self.use_bias:
            out = out + self.b_2.data
        return out

    def parameters(self) -> List[Tensor]:
        if self.activation == ActivationType.SWIGLU:
            params = [self.W_up, self.W_gate, self.W_down]
            if self.use_bias:
                params.extend([self.b_up, self.b_gate, self.b_down])
        else:
            params = [self.W_1, self.W_2]
            if self.use_bias:
                params.extend([self.b_1, self.b_2])
        return params


class Dropout:
    """Dropout layer (training-time only)."""

    def __init__(self, rate: float = 0.1):
        self.rate = rate
        self.training = True

    def forward(self, x: np.ndarray) -> np.ndarray:
        if not self.training or self.rate == 0.0:
            return x
        mask = (np.random.rand(*x.shape) > self.rate).astype(np.float32)
        return x * mask / (1.0 - self.rate)


# ─── Transformer Block ──────────────────────────────────────

class TransformerBlock:
    """
    Complete transformer block with all normalization and
    activation variants. Supports pre-norm and post-norm.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        norm_type: NormType = NormType.RMSNORM,
        norm_placement: NormPlacement = NormPlacement.PRE,
        activation: ActivationType = ActivationType.GELU,
        attention_type: AttentionType = AttentionType.FULL,
        dropout: float = 0.1,
        use_bias: bool = True,
        num_layers: int = 12,
    ):
        self.d_model = d_model
        self.norm_placement = norm_placement

        # Attention sublayer
        self.attention = MultiHeadAttention(d_model, num_heads, attention_type, dropout)

        # Normalization layers
        def make_norm():
            if norm_type == NormType.LAYERNORM:
                return LayerNorm(d_model)
            elif norm_type == NormType.RMSNORM:
                return RMSNorm(d_model)
            elif norm_type == NormType.DEEPNORM:
                return DeepNorm(d_model, num_layers)
            raise ValueError(f"Unknown norm type: {norm_type}")

        self.norm1 = make_norm()
        self.norm2 = make_norm()

        # MLP sublayer
        self.mlp = MLP(d_model, d_ff, activation, use_bias)

        # Dropout
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def forward(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None,
        kv_cache: Optional[tuple] = None,
        store_intermediates: bool = False,
    ) -> TransformerBlockOutput:
        intermediates: Dict[str, np.ndarray] = {}

        if self.norm_placement == NormPlacement.PRE:
            # ── Pre-Norm (GPT-3, LLaMA style) ────────────────
            # Attention sublayer
            normed = self.norm1.forward(x)
            if store_intermediates:
                intermediates["norm1_output"] = normed.copy()

            attn_out, attn_w = self.attention.forward(normed, mask, kv_cache)
            if store_intermediates:
                intermediates["attention_output"] = attn_out.copy()
                intermediates["attention_weights"] = attn_w.copy()

            attn_out = self.dropout1.forward(attn_out)
            x = x + attn_out  # residual
            if store_intermediates:
                intermediates["after_attention_residual"] = x.copy()

            # MLP sublayer
            normed = self.norm2.forward(x)
            if store_intermediates:
                intermediates["norm2_output"] = normed.copy()

            mlp_out = self.mlp.forward(normed)
            if store_intermediates:
                intermediates["mlp_output"] = mlp_out.copy()

            mlp_out = self.dropout2.forward(mlp_out)
            x = x + mlp_out  # residual
            if store_intermediates:
                intermediates["final_output"] = x.copy()

        else:
            # ── Post-Norm (original Transformer style) ────────
            attn_out, attn_w = self.attention.forward(x, mask, kv_cache)
            attn_out = self.dropout1.forward(attn_out)
            x = self.norm1.forward(x + attn_out)

            mlp_out = self.mlp.forward(x)
            mlp_out = self.dropout2.forward(mlp_out)
            x = self.norm2.forward(x + mlp_out)

            if store_intermediates:
                intermediates["attention_weights"] = attn_w.copy()
                intermediates["final_output"] = x.copy()

        return TransformerBlockOutput(
            output=x,
            intermediates=intermediates if store_intermediates else None,
        )

    def parameters(self) -> List[Tensor]:
        params: List[Tensor] = []
        params.extend(self.attention.parameters())
        params.extend(self.norm1.parameters())
        params.extend(self.norm2.parameters())
        params.extend(self.mlp.parameters())
        return params

    def set_training(self, training: bool):
        self.dropout1.training = training
        self.dropout2.training = training
