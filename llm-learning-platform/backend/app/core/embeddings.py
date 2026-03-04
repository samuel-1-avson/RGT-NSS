"""
Embedding Layer System

Token embeddings with multiple positional encoding strategies:
sinusoidal, learned, RoPE, and ALiBi. Includes similarity search,
analogy computation, and geometric analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.core.tensor import Tensor


class InitStrategy(str, Enum):
    XAVIER = "xavier"
    HE = "he"
    NORMAL = "normal"


class PositionalEncodingType(str, Enum):
    SINUSOIDAL = "sinusoidal"
    LEARNED = "learned"
    ROPE = "rope"
    ALIBI = "alibi"
    NONE = "none"


@dataclass
class EmbeddingGeometry:
    isotropy: float
    effective_dimensionality: float
    eigenvalue_spectrum: List[float]
    mean_norm: float
    std_norm: float


@dataclass
class SimilarityResult:
    token_id: int
    token: str
    score: float


# ─── Embedding Layer ─────────────────────────────────────────

class EmbeddingLayer:
    """
    Comprehensive embedding layer with visualization and analysis
    support. Implements token + positional embeddings.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        max_seq_len: int = 2048,
        init_strategy: InitStrategy = InitStrategy.NORMAL,
        positional_encoding: PositionalEncodingType = PositionalEncodingType.SINUSOIDAL,
        dropout: float = 0.1,
    ):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len

        # Token embeddings
        self.weight = Tensor(
            self._initialize_weights(vocab_size, embedding_dim, init_strategy),
            requires_grad=True,
        )

        # Positional encodings
        self.pos_type = positional_encoding
        if positional_encoding == PositionalEncodingType.SINUSOIDAL:
            self.pos_emb = self._create_sinusoidal(max_seq_len, embedding_dim)
        elif positional_encoding == PositionalEncodingType.LEARNED:
            self.pos_emb = Tensor(
                np.random.randn(max_seq_len, embedding_dim).astype(np.float32) * 0.02,
                requires_grad=True,
            )
        elif positional_encoding == PositionalEncodingType.ROPE:
            self.rope_cos, self.rope_sin = self._precompute_rope(
                max_seq_len, embedding_dim
            )
        # ALiBi doesn't add positional embeddings to tokens; it biases attention

        self.dropout_rate = dropout

        # Token mappings (set externally)
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

    # ─── Initialization ──────────────────────────────────────

    @staticmethod
    def _initialize_weights(
        rows: int, cols: int, strategy: InitStrategy
    ) -> np.ndarray:
        if strategy == InitStrategy.XAVIER:
            limit = np.sqrt(6.0 / (rows + cols))
            return np.random.uniform(-limit, limit, (rows, cols)).astype(np.float32)
        elif strategy == InitStrategy.HE:
            std = np.sqrt(2.0 / rows)
            return (np.random.randn(rows, cols) * std).astype(np.float32)
        else:  # NORMAL
            return (np.random.randn(rows, cols) * 0.02).astype(np.float32)

    @staticmethod
    def _create_sinusoidal(max_len: int, dim: int) -> np.ndarray:
        """Create sinusoidal positional encodings (Vaswani et al.)."""
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(
            np.arange(0, dim, 2) * -(np.log(10000.0) / dim)
        )
        pe = np.zeros((max_len, dim), dtype=np.float32)
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return pe

    @staticmethod
    def _precompute_rope(
        max_len: int, dim: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Precompute Rotary Position Embedding (RoPE) cache."""
        inv_freq = 1.0 / (
            10000 ** (np.arange(0, dim, 2, dtype=np.float32) / dim)
        )
        t = np.arange(max_len, dtype=np.float32)
        freqs = np.outer(t, inv_freq)
        emb = np.concatenate([freqs, freqs], axis=-1)
        return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)

    # ─── Forward Pass ────────────────────────────────────────

    def forward(
        self,
        token_ids: np.ndarray,
        positions: Optional[np.ndarray] = None,
    ) -> Tensor:
        """
        Embed tokens and add positional information.

        Args:
            token_ids: (batch, seq_len) integer token IDs
            positions: optional explicit position indices
        Returns:
            Tensor of shape (batch, seq_len, embedding_dim)
        """
        # Token embeddings via lookup
        token_emb = Tensor(
            self.weight.data[token_ids],
            requires_grad=True,
        )

        seq_len = token_ids.shape[-1] if token_ids.ndim > 1 else token_ids.shape[0]
        if positions is None:
            positions = np.arange(seq_len)

        if self.pos_type == PositionalEncodingType.SINUSOIDAL:
            pe = self.pos_emb[positions]
            return token_emb + Tensor(pe)

        elif self.pos_type == PositionalEncodingType.LEARNED:
            pe = Tensor(self.pos_emb.data[positions], requires_grad=True)
            return token_emb + pe

        elif self.pos_type == PositionalEncodingType.ROPE:
            # RoPE is applied in the attention layer, not here
            return token_emb

        else:
            return token_emb

    def apply_rope(self, x: np.ndarray, seq_len: int) -> np.ndarray:
        """Apply Rotary Position Embeddings to Q or K."""
        cos = self.rope_cos[:seq_len]
        sin = self.rope_sin[:seq_len]

        # Split x into pairs and rotate
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        cos_part = cos[:, : x1.shape[-1]]
        sin_part = sin[:, : x1.shape[-1]]

        rotated = np.concatenate(
            [x1 * cos_part - x2 * sin_part, x1 * sin_part + x2 * cos_part],
            axis=-1,
        )
        return rotated.astype(np.float32)

    @staticmethod
    def compute_alibi_bias(num_heads: int, max_len: int) -> np.ndarray:
        """Compute ALiBi attention bias matrix."""
        slopes = np.array(
            [2 ** (-8 * i / num_heads) for i in range(1, num_heads + 1)],
            dtype=np.float32,
        )
        positions = np.arange(max_len, dtype=np.float32)
        # bias[h, i, j] = slope_h * (j - i)  (causal: only j <= i)
        rel_pos = positions[None, :] - positions[:, None]  # (seq, seq)
        bias = slopes[:, None, None] * rel_pos[None, :, :]  # (heads, seq, seq)
        return bias

    # ─── Parameters ──────────────────────────────────────────

    def parameters(self) -> List[Tensor]:
        params = [self.weight]
        if self.pos_type == PositionalEncodingType.LEARNED:
            params.append(self.pos_emb)
        return params

    # ─── Similarity & Analysis ───────────────────────────────

    def get_similar_tokens(
        self,
        token_id: int,
        k: int = 10,
        metric: str = "cosine",
    ) -> List[SimilarityResult]:
        """Find k most similar tokens by embedding distance."""
        token_emb = self.weight.data[token_id]

        if metric == "cosine":
            norms = np.linalg.norm(self.weight.data, axis=1, keepdims=True) + 1e-8
            normed = self.weight.data / norms
            token_normed = token_emb / (np.linalg.norm(token_emb) + 1e-8)
            similarities = normed @ token_normed
        else:
            distances = np.linalg.norm(self.weight.data - token_emb, axis=1)
            similarities = -distances

        top_ids = np.argsort(similarities)[-(k + 1) : -1][::-1]
        return [
            SimilarityResult(
                token_id=int(idx),
                token=self.id_to_token.get(int(idx), f"<{idx}>"),
                score=float(similarities[idx]),
            )
            for idx in top_ids
        ]

    def compute_analogy(
        self,
        a: int,
        b: int,
        c: int,
        k: int = 5,
    ) -> List[SimilarityResult]:
        """Solve analogy: a is to b as c is to ?"""
        result_vec = self.weight.data[b] - self.weight.data[a] + self.weight.data[c]
        norms = np.linalg.norm(self.weight.data, axis=1, keepdims=True) + 1e-8
        normed = self.weight.data / norms
        result_normed = result_vec / (np.linalg.norm(result_vec) + 1e-8)
        similarities = normed @ result_normed

        exclude = {a, b, c}
        top_ids = np.argsort(similarities)[::-1]
        results = []
        for idx in top_ids:
            if int(idx) not in exclude:
                results.append(
                    SimilarityResult(
                        token_id=int(idx),
                        token=self.id_to_token.get(int(idx), f"<{idx}>"),
                        score=float(similarities[idx]),
                    )
                )
            if len(results) >= k:
                break
        return results

    def analyze_geometry(self) -> EmbeddingGeometry:
        """Analyze geometric properties of the embedding space."""
        W = self.weight.data
        centered = W - W.mean(axis=0)
        cov = (centered.T @ centered) / len(W)
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.sort(eigenvalues)[::-1]

        isotropy = float(eigenvalues[0] / (eigenvalues[-1] + 1e-8))
        effective_dim = float(
            np.sum(eigenvalues) ** 2 / (np.sum(eigenvalues ** 2) + 1e-8)
        )
        norms = np.linalg.norm(W, axis=1)

        return EmbeddingGeometry(
            isotropy=isotropy,
            effective_dimensionality=effective_dim,
            eigenvalue_spectrum=eigenvalues[:50].tolist(),
            mean_norm=float(norms.mean()),
            std_norm=float(norms.std()),
        )

    def reduce_dimensions(
        self,
        method: str = "pca",
        n_components: int = 2,
    ) -> np.ndarray:
        """Project embeddings to lower dimensions for visualization."""
        W = self.weight.data
        centered = W - W.mean(axis=0)

        if method == "pca":
            cov = (centered.T @ centered) / len(W)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            idx = np.argsort(eigenvalues)[::-1][:n_components]
            components = eigenvectors[:, idx]
            return (centered @ components).astype(np.float32)
        else:
            # Fallback to PCA
            return self.reduce_dimensions("pca", n_components)
