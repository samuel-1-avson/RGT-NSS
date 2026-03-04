"""
Attention Mechanism System

Full, local, sparse, and linear attention — all with step-by-step
visualization support for educational transparency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.core.tensor import Tensor


class AttentionType(str, Enum):
    FULL = "full"
    LOCAL = "local"
    SPARSE = "sparse"
    LINEAR = "linear"


@dataclass
class AttentionStepResult:
    output: np.ndarray
    attention_weights: np.ndarray
    intermediates: Dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class AttentionPatternAnalysis:
    patterns: Dict[str, dict]


# ─── Utility Functions ───────────────────────────────────────

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - x.max(axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / exp_x.sum(axis=axis, keepdims=True)


# ─── Multi-Head Attention ────────────────────────────────────

class MultiHeadAttention:
    """
    Complete multi-head attention with all variants and
    step-by-step visualization support.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        attention_type: AttentionType = AttentionType.FULL,
        dropout: float = 0.0,
        window_size: int = 64,
    ):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.attention_type = attention_type
        self.dropout = dropout
        self.window_size = window_size

        scale = np.sqrt(2.0 / d_model).astype(np.float32)
        self.W_q = Tensor(np.random.randn(d_model, d_model).astype(np.float32) * scale, requires_grad=True)
        self.W_k = Tensor(np.random.randn(d_model, d_model).astype(np.float32) * scale, requires_grad=True)
        self.W_v = Tensor(np.random.randn(d_model, d_model).astype(np.float32) * scale, requires_grad=True)
        self.W_o = Tensor(np.random.randn(d_model, d_model).astype(np.float32) * scale, requires_grad=True)

        self.b_q = Tensor(np.zeros(d_model, dtype=np.float32), requires_grad=True)
        self.b_k = Tensor(np.zeros(d_model, dtype=np.float32), requires_grad=True)
        self.b_v = Tensor(np.zeros(d_model, dtype=np.float32), requires_grad=True)
        self.b_o = Tensor(np.zeros(d_model, dtype=np.float32), requires_grad=True)

    def parameters(self) -> List[Tensor]:
        return [
            self.W_q, self.W_k, self.W_v, self.W_o,
            self.b_q, self.b_k, self.b_v, self.b_o,
        ]

    def forward(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None,
        kv_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass returning (output, attention_weights).
        """
        result = self.compute_step_by_step(x, mask, kv_cache, store_intermediates=False)
        return result.output, result.attention_weights

    def compute_step_by_step(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None,
        kv_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        store_intermediates: bool = True,
    ) -> AttentionStepResult:
        """Compute attention with full step-by-step data for visualization."""
        intermediates: Dict[str, np.ndarray] = {}

        if x.ndim == 2:
            x = x[np.newaxis, :, :]  # add batch dim

        batch_size, seq_len, _ = x.shape

        # Step 1: Linear projections
        Q = x @ self.W_q.data + self.b_q.data
        K = x @ self.W_k.data + self.b_k.data
        V = x @ self.W_v.data + self.b_v.data

        if store_intermediates:
            intermediates["Q_projected"] = Q.copy()
            intermediates["K_projected"] = K.copy()
            intermediates["V_projected"] = V.copy()

        # Use KV cache if provided
        if kv_cache is not None:
            K = np.concatenate([kv_cache[0], K], axis=1)
            V = np.concatenate([kv_cache[1], V], axis=1)

        kv_len = K.shape[1]

        # Step 2: Reshape for multi-head
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.d_head).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, kv_len, self.num_heads, self.d_head).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, kv_len, self.num_heads, self.d_head).transpose(0, 2, 1, 3)

        if store_intermediates:
            intermediates["Q_heads"] = Q.copy()
            intermediates["K_heads"] = K.copy()
            intermediates["V_heads"] = V.copy()

        # Step 3: Compute attention
        if self.attention_type == AttentionType.FULL:
            attn_out, attn_weights = self._full_attention(Q, K, V, mask)
        elif self.attention_type == AttentionType.LOCAL:
            attn_out, attn_weights = self._local_attention(Q, K, V)
        elif self.attention_type == AttentionType.SPARSE:
            attn_out, attn_weights = self._sparse_attention(Q, K, V, mask)
        elif self.attention_type == AttentionType.LINEAR:
            attn_out, attn_weights = self._linear_attention(Q, K, V)
        else:
            attn_out, attn_weights = self._full_attention(Q, K, V, mask)

        if store_intermediates:
            intermediates["attention_weights"] = attn_weights.copy()

        # Step 4: Merge heads
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)

        # Step 5: Output projection
        output = attn_out @ self.W_o.data + self.b_o.data

        if store_intermediates:
            intermediates["final_output"] = output.copy()

        return AttentionStepResult(
            output=output,
            attention_weights=attn_weights,
            intermediates=intermediates,
        )

    # ─── Attention Variants ──────────────────────────────────

    def _full_attention(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Standard scaled dot-product attention."""
        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.d_head)

        if mask is not None:
            scores = scores + mask

        attn_weights = softmax(scores, axis=-1)
        output = attn_weights @ V
        return output, attn_weights

    def _local_attention(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Local windowed attention for long sequences."""
        batch_size, num_heads, seq_len, d_head = Q.shape
        kv_len = K.shape[2]

        output = np.zeros_like(Q)
        attn_weights = np.zeros((batch_size, num_heads, seq_len, kv_len), dtype=np.float32)

        half_w = self.window_size // 2
        for i in range(seq_len):
            start = max(0, i - half_w)
            end = min(kv_len, i + half_w + 1)

            q_i = Q[:, :, i : i + 1, :]
            k_win = K[:, :, start:end, :]
            v_win = V[:, :, start:end, :]

            scores = q_i @ k_win.transpose(0, 1, 3, 2) / np.sqrt(self.d_head)
            local_attn = softmax(scores, axis=-1)

            output[:, :, i : i + 1, :] = local_attn @ v_win
            attn_weights[:, :, i : i + 1, start:end] = local_attn

        return output, attn_weights

    def _sparse_attention(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sparse attention with local + global tokens (Longformer-style)."""
        batch_size, num_heads, seq_len, d_head = Q.shape
        kv_len = K.shape[2]

        # Create sparse mask: local window + first/last token global
        sparse_mask = np.full((seq_len, kv_len), -1e9, dtype=np.float32)
        half_w = self.window_size // 2
        for i in range(seq_len):
            start = max(0, i - half_w)
            end = min(kv_len, i + half_w + 1)
            sparse_mask[i, start:end] = 0.0
            sparse_mask[i, 0] = 0.0  # first token is global
            if kv_len > 1:
                sparse_mask[i, -1] = 0.0  # last token is global

        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.d_head)
        scores = scores + sparse_mask[np.newaxis, np.newaxis, :, :]

        if mask is not None:
            scores = scores + mask

        attn_weights = softmax(scores, axis=-1)
        output = attn_weights @ V
        return output, attn_weights

    def _linear_attention(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Linear attention approximation — O(n) complexity."""
        # Feature map: ELU + 1
        Q_prime = np.maximum(Q, 0) + 1
        K_prime = np.maximum(K, 0) + 1

        # KV = K^T V, then Q @ KV
        KV = K_prime.transpose(0, 1, 3, 2) @ V  # (batch, heads, d_head, d_head)
        numerator = Q_prime @ KV
        denominator = Q_prime @ K_prime.transpose(0, 1, 3, 2).sum(axis=-1, keepdims=True)

        output = numerator / (denominator + 1e-8)

        # Approximate weights for visualization
        attn_weights = softmax(Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.d_head), axis=-1)

        return output, attn_weights

    # ─── Analysis ────────────────────────────────────────────

    def analyze_attention_patterns(
        self,
        attention_weights: np.ndarray,
        tokens: Optional[List[str]] = None,
    ) -> AttentionPatternAnalysis:
        """Analyze attention patterns for interpretability."""
        if attention_weights.ndim == 3:
            attention_weights = attention_weights[np.newaxis, :]

        batch_size, num_heads, seq_len, _ = attention_weights.shape
        patterns: Dict[str, dict] = {}

        for head in range(num_heads):
            head_attn = attention_weights[0, head]

            # Diagonal (local) focus
            diag_score = float(np.mean(np.diag(head_attn)))

            # Vertical (positional) focus
            col_means = head_attn.mean(axis=0)
            vertical_score = float(np.max(col_means))

            # Entropy (uniformity)
            entropy = float(
                -np.sum(head_attn * np.log(head_attn + 1e-10), axis=-1).mean()
            )

            # Sparsity
            sparsity = float(np.mean(np.sum(head_attn > 0.1, axis=-1)))

            pattern_type = self._classify_pattern(diag_score, vertical_score, entropy)

            patterns[f"head_{head}"] = {
                "diagonal_focus": diag_score,
                "vertical_focus": vertical_score,
                "entropy": entropy,
                "sparsity": sparsity,
                "pattern_type": pattern_type,
            }

        return AttentionPatternAnalysis(patterns=patterns)

    @staticmethod
    def _classify_pattern(
        diagonal: float, vertical: float, entropy: float
    ) -> str:
        if diagonal > 0.3:
            return "local/diagonal"
        elif vertical > 0.5:
            return "vertical/position"
        elif entropy < 1.0:
            return "sparse/concentrated"
        else:
            return "distributed/global"

    @staticmethod
    def create_causal_mask(seq_len: int) -> np.ndarray:
        """Create causal (autoregressive) attention mask."""
        mask = np.triu(np.ones((seq_len, seq_len), dtype=np.float32), k=1)
        return (mask * -1e9).astype(np.float32)

    @staticmethod
    def create_padding_mask(lengths: np.ndarray, max_len: int) -> np.ndarray:
        """Create padding mask from sequence lengths."""
        batch_size = len(lengths)
        mask = np.zeros((batch_size, 1, 1, max_len), dtype=np.float32)
        for i, length in enumerate(lengths):
            mask[i, 0, 0, int(length) :] = -1e9
        return mask
