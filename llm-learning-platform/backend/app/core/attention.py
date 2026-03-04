"""
Attention Engine — PyTorch GPU-accelerated Multi-Head Attention

Real scaled dot-product attention with causal masking,
multiple attention variants, and step-by-step visualization support.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.core.device import get_device


class AttentionType(Enum):
    FULL = "full"
    LOCAL = "local"
    SPARSE = "sparse"
    LINEAR = "linear"


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention as a real nn.Module with GPU acceleration.

    Supports full, local-window, sparse, and linear attention variants.
    Stores attention weights for visualization.
    """

    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        attention_type: AttentionType = AttentionType.FULL,
        local_window: int = 32,
        use_bias: bool = True,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.attention_type = attention_type
        self.local_window = local_window

        self.q_proj = nn.Linear(d_model, d_model, bias=use_bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=use_bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=use_bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=use_bias)

        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

        self._last_attention_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        store_weights: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (seq_len, seq_len) causal mask
            store_weights: save attention weights for visualization
        """
        B, S, D = x.shape

        Q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply attention type masking
        if self.attention_type == AttentionType.LOCAL:
            local_mask = self._create_local_mask(S).to(x.device)
            scores = scores.masked_fill(local_mask == 0, float("-inf"))
        elif self.attention_type == AttentionType.SPARSE:
            sparse_mask = self._create_sparse_mask(S).to(x.device)
            scores = scores.masked_fill(sparse_mask == 0, float("-inf"))

        # Causal mask
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 0, float("-inf"))

        if self.attention_type == AttentionType.LINEAR:
            # Linear attention: kernel approximation (ELU + 1)
            Q_prime = F.elu(Q) + 1
            K_prime = F.elu(K) + 1
            KV = torch.matmul(K_prime.transpose(-2, -1), V)
            output = torch.matmul(Q_prime, KV)
            denom = torch.matmul(Q_prime, K_prime.transpose(-2, -1).sum(dim=-1, keepdim=True))
            output = output / (denom + 1e-6)
            attn_weights = None
        else:
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            output = torch.matmul(attn_weights, V)

        if store_weights and attn_weights is not None:
            self._last_attention_weights = attn_weights.detach()

        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(B, S, D)
        output = self.out_proj(output)
        return self.dropout(output)

    def _create_local_mask(self, seq_len: int) -> torch.Tensor:
        mask = torch.zeros(seq_len, seq_len)
        for i in range(seq_len):
            start = max(0, i - self.local_window)
            end = min(seq_len, i + self.local_window + 1)
            mask[i, start:end] = 1
        return mask

    def _create_sparse_mask(self, seq_len: int, stride: int = 4) -> torch.Tensor:
        mask = torch.zeros(seq_len, seq_len)
        for i in range(seq_len):
            # Local window
            start = max(0, i - self.local_window)
            end = min(seq_len, i + 1)
            mask[i, start:end] = 1
            # Strided global attention
            for j in range(0, seq_len, stride):
                if j <= i:
                    mask[i, j] = 1
        return mask

    @staticmethod
    def create_causal_mask(seq_len: int) -> torch.Tensor:
        """Create a causal (autoregressive) attention mask."""
        return torch.tril(torch.ones(seq_len, seq_len, device=get_device()))

    def get_attention_weights(self) -> Optional[np.ndarray]:
        """Return last stored attention weights as numpy array."""
        if self._last_attention_weights is not None:
            return self._last_attention_weights.cpu().numpy()
        return None

    def forward_step_by_step(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict:
        """Run attention with full step-by-step intermediate outputs for visualization."""
        B, S, D = x.shape

        Q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(B, S, D)
        output = self.out_proj(output)

        return {
            "queries": Q[0].detach().cpu().numpy().tolist(),
            "keys": K[0].detach().cpu().numpy().tolist(),
            "values": V[0].detach().cpu().numpy().tolist(),
            "scores": scores[0].detach().cpu().numpy().tolist(),
            "attention_weights": attn_weights[0].detach().cpu().numpy().tolist(),
            "output": output[0].detach().cpu().numpy().tolist(),
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "seq_len": S,
        }

    def analyze_attention_patterns(self, x: torch.Tensor) -> Dict:
        """Analyze attention patterns across all heads."""
        mask = self.create_causal_mask(x.size(1))
        self.forward(x, mask=mask, store_weights=True)
        weights = self._last_attention_weights  # (B, H, S, S)

        if weights is None:
            return {"error": "No attention weights available"}

        w = weights[0]  # First batch
        patterns = []
        for h in range(self.num_heads):
            head_w = w[h]
            entropy = -(head_w * (head_w + 1e-8).log()).sum(dim=-1).mean().item()
            sparsity = (head_w < 0.01).float().mean().item()
            patterns.append({
                "head": h,
                "avg_entropy": round(entropy, 4),
                "sparsity": round(sparsity, 4),
                "max_attention": round(head_w.max().item(), 4),
            })

        return {
            "num_heads": self.num_heads,
            "seq_len": x.size(1),
            "attention_type": self.attention_type.value,
            "head_patterns": patterns,
            "attention_matrix": w.cpu().numpy().round(4).tolist(),
        }
