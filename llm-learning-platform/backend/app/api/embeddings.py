"""
Embeddings API

Explore word embeddings, positional encodings, similarity search,
analogy computation, and geometric analysis.
"""

from typing import List, Optional

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.core.embeddings import EmbeddingLayer, PositionalEncodingType

router = APIRouter()


class EmbeddingRequest(BaseModel):
    token_ids: List[int] = Field(default=[1, 5, 10, 20, 30])
    d_model: int = Field(default=64, ge=8, le=1024)
    vocab_size: int = Field(default=256, ge=10, le=100000)
    positional_encoding: str = "sinusoidal"
    max_seq_len: int = 128


class SimilarityRequest(BaseModel):
    query_id: int = 1
    d_model: int = 64
    vocab_size: int = 256
    top_k: int = 10
    metric: str = "cosine"


class AnalogyRequest(BaseModel):
    a: int
    b: int
    c: int
    d_model: int = 64
    vocab_size: int = 256
    top_k: int = 5


@router.post("/encode")
async def encode_tokens(request: EmbeddingRequest):
    """Get embeddings for token IDs with positional encoding."""
    pe_type = PositionalEncodingType(request.positional_encoding)
    layer = EmbeddingLayer(
        vocab_size=request.vocab_size,
        embedding_dim=request.d_model,
        max_seq_len=request.max_seq_len,
        positional_encoding=pe_type,
    )

    ids = np.array([request.token_ids])
    with_pos = layer.forward(ids).data
    token_emb = layer.weight.data[ids]

    return {
        "token_embeddings_shape": list(token_emb.shape),
        "with_position_shape": list(with_pos.shape),
        "token_embeddings": token_emb[0].tolist(),
        "with_positional": with_pos[0].tolist(),
        "positional_encoding_type": request.positional_encoding,
    }


@router.post("/similarity")
async def find_similar(request: SimilarityRequest):
    """Find most similar tokens by embedding distance."""
    layer = EmbeddingLayer(
        vocab_size=request.vocab_size,
        embedding_dim=request.d_model,
    )

    results = layer.get_similar_tokens(
        request.query_id,
        k=request.top_k,
        metric=request.metric,
    )

    return {
        "query_id": request.query_id,
        "metric": request.metric,
        "similar_tokens": [
            {"token_id": int(item.token_id), "token": item.token, "score": float(item.score)}
            for item in results
        ],
    }


@router.post("/analogy")
async def compute_analogy(request: AnalogyRequest):
    """Compute word analogy: a is to b as c is to ?"""
    layer = EmbeddingLayer(
        vocab_size=request.vocab_size,
        embedding_dim=request.d_model,
    )

    results = layer.compute_analogy(request.a, request.b, request.c, k=request.top_k)

    return {
        "analogy": f"{request.a} : {request.b} :: {request.c} : ?",
        "results": [
            {"token_id": int(item.token_id), "token": item.token, "score": float(item.score)}
            for item in results
        ],
    }


@router.post("/geometry")
async def analyze_geometry(request: EmbeddingRequest):
    """Analyze embedding space geometry."""
    layer = EmbeddingLayer(
        vocab_size=request.vocab_size,
        embedding_dim=request.d_model,
    )

    analysis = layer.analyze_geometry()

    return {
        "token_ids": request.token_ids,
        "mean_norm": float(analysis.mean_norm),
        "std_norm": float(analysis.std_norm),
        "isotropy_score": float(analysis.isotropy),
        "effective_dimensionality": float(analysis.effective_dimensionality),
        "eigenvalue_spectrum": analysis.eigenvalue_spectrum[:10],
    }


@router.get("/positional-encodings")
async def get_positional_encoding_types():
    """List available positional encoding types."""
    return {
        "types": [
            {
                "name": pe.value,
                "description": {
                    "sinusoidal": "Fixed sinusoidal encodings (Vaswani et al., 2017)",
                    "learned": "Learned positional embeddings",
                    "rope": "Rotary Position Embedding (Su et al., 2021)",
                    "alibi": "Attention with Linear Biases (Press et al., 2022)",
                }.get(pe.value, ""),
            }
            for pe in PositionalEncodingType
        ]
    }
