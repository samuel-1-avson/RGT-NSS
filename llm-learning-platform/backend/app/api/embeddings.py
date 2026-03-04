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
    
    import torch
    weights = layer.token_embedding.weight.data
    query_vec = weights[request.query_id].unsqueeze(0)
    
    if request.metric == "cosine":
        sim = torch.nn.functional.cosine_similarity(query_vec, weights)
    else:
        # l2 distance converted to a similarity score format
        sim = -torch.cdist(query_vec, weights).squeeze()
        
    top_indices = torch.topk(sim, k=request.top_k + 1).indices
    top_indices = [idx.item() for idx in top_indices if idx.item() != request.query_id][:request.top_k]
    scores = sim[top_indices].tolist()

    return {
        "query_id": request.query_id,
        "metric": request.metric,
        "similar_tokens": [
            {"token_id": idx, "token": f"token_{idx}", "score": float(score)}
            for idx, score in zip(top_indices, scores)
        ],
    }


@router.post("/analogy")
async def compute_analogy(request: AnalogyRequest):
    """Compute word analogy: a is to b as c is to ?"""
    layer = EmbeddingLayer(
        vocab_size=request.vocab_size,
        embedding_dim=request.d_model,
    )
    import torch

    weights = layer.token_embedding.weight.data
    # a is to b as c is to d => d = b - a + c
    vec_a = weights[request.a]
    vec_b = weights[request.b]
    vec_c = weights[request.c]
    
    target_vec = (vec_b - vec_a + vec_c).unsqueeze(0)
    sim = torch.nn.functional.cosine_similarity(target_vec, weights)
    
    # Exclude input tokens
    exclude = {request.a, request.b, request.c}
    top_indices = []
    scores = []
    
    for idx in torch.argsort(sim, descending=True):
        if idx.item() not in exclude:
            top_indices.append(idx.item())
            scores.append(sim[idx].item())
        if len(top_indices) >= request.top_k:
            break

    return {
        "analogy": f"{request.a} : {request.b} :: {request.c} : ?",
        "results": [
            {"token_id": idx, "token": f"token_{idx}", "score": float(score)}
            for idx, score in zip(top_indices, scores)
        ],
    }


@router.post("/geometry")
async def analyze_geometry(request: EmbeddingRequest):
    """Analyze embedding space geometry."""
    layer = EmbeddingLayer(
        vocab_size=request.vocab_size,
        embedding_dim=request.d_model,
    )
    import torch
    
    weights = layer.token_embedding.weight.data
    norms = torch.norm(weights, dim=1)
    mean_norm = norms.mean().item()
    std_norm = norms.std().item()
    
    # Simple isometry score logic for demo
    centered = weights - weights.mean(dim=0)
    cov = (centered.T @ centered) / weights.shape[0]
    eigenvalues = torch.linalg.eigvalsh(cov).real
    
    # Avoid div zero
    total_var = eigenvalues.sum().clamp_min(1e-9)
    isotropy_score = (eigenvalues.min() / eigenvalues.max()).item() if eigenvalues.max() > 0 else 0
    effective_dimensionality = (total_var**2 / (eigenvalues**2).sum().clamp_min(1e-9)).item()

    return {
        "token_ids": request.token_ids,
        "mean_norm": float(mean_norm),
        "std_norm": float(std_norm),
        "isotropy_score": float(isotropy_score),
        "effective_dimensionality": float(effective_dimensionality),
        "eigenvalue_spectrum": eigenvalues.tolist()[:10],
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
