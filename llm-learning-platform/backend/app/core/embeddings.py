"""
Embedding Engine — PyTorch GPU-accelerated Embeddings

Token embeddings + positional encodings (sinusoidal, learned, RoPE)
as real nn.Module components with GPU placement.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from app.core.device import get_device


class PositionalEncodingType(Enum):
    SINUSOIDAL = "sinusoidal"
    LEARNED = "learned"
    ROPE = "rope"
    NONE = "none"


class InitStrategy(Enum):
    NORMAL = "normal"
    XAVIER = "xavier"
    KAIMING = "kaiming"


class EmbeddingLayer(nn.Module):
    """
    Token + positional embedding layer with multiple encoding strategies.

    Fully GPU-accelerated via PyTorch nn.Module.
    """

    def __init__(
        self,
        vocab_size: int = 256,
        embedding_dim: int = 128,
        max_seq_len: int = 256,
        init_strategy: InitStrategy = InitStrategy.NORMAL,
        positional_encoding: PositionalEncodingType = PositionalEncodingType.SINUSOIDAL,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.pos_type = positional_encoding

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self._init_weights(init_strategy)

        # Positional encoding
        if positional_encoding == PositionalEncodingType.SINUSOIDAL:
            pe = self._build_sinusoidal(max_seq_len, embedding_dim)
            self.register_buffer("pos_encoding", pe)
        elif positional_encoding == PositionalEncodingType.LEARNED:
            self.pos_embedding = nn.Embedding(max_seq_len, embedding_dim)
        # RoPE is applied in attention, not here

        self.dropout = nn.Dropout(dropout)

    def _init_weights(self, strategy: InitStrategy):
        if strategy == InitStrategy.NORMAL:
            nn.init.normal_(self.token_embedding.weight, std=0.02)
        elif strategy == InitStrategy.XAVIER:
            nn.init.xavier_uniform_(self.token_embedding.weight)
        elif strategy == InitStrategy.KAIMING:
            nn.init.kaiming_uniform_(self.token_embedding.weight)

    @staticmethod
    def _build_sinusoidal(max_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        return pe.unsqueeze(0)  # (1, max_len, d_model)

    @property
    def weight(self):
        """Backward-compatible access to embedding weight."""
        return self.token_embedding.weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: (batch, seq_len) integer token IDs
        Returns:
            (batch, seq_len, embedding_dim) embeddings
        """
        if isinstance(token_ids, np.ndarray):
            token_ids = torch.from_numpy(token_ids).long().to(get_device())

        x = self.token_embedding(token_ids)  # (B, S, D)

        if self.pos_type == PositionalEncodingType.SINUSOIDAL:
            seq_len = x.size(1)
            x = x + self.pos_encoding[:, :seq_len, :]
        elif self.pos_type == PositionalEncodingType.LEARNED:
            positions = torch.arange(x.size(1), device=x.device)
            x = x + self.pos_embedding(positions)

        return self.dropout(x)

    def parameters_list(self) -> List:
        """Return parameters for the custom Tensor-based API."""
        return list(self.parameters())

    def get_embedding_info(self) -> Dict:
        return {
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "max_seq_len": self.max_seq_len,
            "positional_encoding": self.pos_type.value,
            "total_params": sum(p.numel() for p in self.parameters()),
            "device": str(next(self.parameters()).device),
        }

    def encode_tokens(self, token_ids: List[int]) -> Dict:
        """Encode tokens and return embeddings with metadata."""
        ids = torch.tensor([token_ids], dtype=torch.long, device=get_device())
        with torch.no_grad():
            embeddings = self.forward(ids)
        return {
            "token_ids": token_ids,
            "embeddings": embeddings[0].cpu().numpy().tolist(),
            "shape": list(embeddings.shape),
            "embedding_dim": self.embedding_dim,
        }

    def compute_similarity(self, id_a: int, id_b: int) -> Dict:
        """Compute cosine similarity between two token embeddings."""
        with torch.no_grad():
            emb_a = self.token_embedding.weight[id_a]
            emb_b = self.token_embedding.weight[id_b]
            cos_sim = torch.nn.functional.cosine_similarity(
                emb_a.unsqueeze(0), emb_b.unsqueeze(0)
            ).item()
        return {
            "token_a": id_a,
            "token_b": id_b,
            "cosine_similarity": round(cos_sim, 6),
            "embedding_dim": self.embedding_dim,
        }

    def compute_geometry(self, token_ids: List[int]) -> Dict:
        """Compute pairwise distances and PCA projection."""
        with torch.no_grad():
            embeddings = self.token_embedding.weight[token_ids]  # (N, D)
            # Pairwise cosine similarity
            normed = torch.nn.functional.normalize(embeddings, dim=1)
            sim_matrix = (normed @ normed.T).cpu().numpy()
            # Simple 2D PCA projection
            centered = embeddings - embeddings.mean(dim=0)
            U, S, V = torch.svd(centered)
            proj_2d = (centered @ V[:, :2]).cpu().numpy()

        return {
            "token_ids": token_ids,
            "similarity_matrix": sim_matrix.round(4).tolist(),
            "pca_2d": proj_2d.round(4).tolist(),
            "embedding_dim": self.embedding_dim,
        }
