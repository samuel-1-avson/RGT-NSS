"""
Standalone computation endpoints for the learning modules.
These do NOT require a pre-trained model — they create temporary lightweight
models on-the-fly to demonstrate real computations.
"""

import os
import numpy as np
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field

from app.models.gpt import MicroGPT, GPTConfig, MultiHeadAttention, RMSNorm
from app.core.tensor import Tensor
from app.core.module import Embedding


router = APIRouter(prefix="/api/compute", tags=["compute"])


# ============== Request/Response Models ==============

class AttentionComputeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=500)
    d_model: int = Field(default=64, ge=16, le=256)
    num_heads: int = Field(default=4, ge=1, le=16)
    num_layers: int = Field(default=1, ge=1, le=4)
    show_causal_mask: bool = True


class EmbeddingComputeRequest(BaseModel):
    text: str = Field(default="", max_length=500)
    vocab_size: int = Field(default=128, ge=16, le=512)
    embedding_dim: int = Field(default=64, ge=8, le=256)
    seed: int = Field(default=42, ge=0, le=99999)


class SamplingRequest(BaseModel):
    logits: Optional[List[float]] = None
    text: Optional[str] = None
    model_id: Optional[str] = None
    temperature: float = Field(default=1.0, ge=0.01, le=5.0)
    top_k: int = Field(default=40, ge=1, le=500)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    vocab_size: int = Field(default=128, ge=16, le=512)


class ForwardStepRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=200)
    d_model: int = Field(default=64, ge=16, le=256)
    num_heads: int = Field(default=4, ge=1, le=16)
    step: int = Field(default=0, ge=0, le=7)


# ============== Helpers ==============

def _char_tokenize(text: str) -> List[int]:
    """Character-level tokenization."""
    return [ord(c) % 256 for c in text]


def _simple_pca_2d(data: np.ndarray) -> np.ndarray:
    """Simple PCA projection to 2D for visualization."""
    mean = np.mean(data, axis=0)
    centered = data - mean
    
    if centered.shape[0] < 2 or centered.shape[1] < 2:
        return centered[:, :2] if centered.shape[1] >= 2 else np.pad(centered, ((0, 0), (0, 2 - centered.shape[1])))
    
    cov = np.cov(centered.T)
    
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]
        projected = centered @ eigenvectors[:, :2]
    except np.linalg.LinAlgError:
        projected = centered[:, :2]
    
    return projected


def _simple_pca_3d(data: np.ndarray) -> np.ndarray:
    """PCA projection to 3D for 3D scatter visualization."""
    mean = np.mean(data, axis=0)
    centered = data - mean
    
    if centered.shape[0] < 3 or centered.shape[1] < 3:
        cols = centered.shape[1]
        if cols >= 3:
            return centered[:, :3]
        return np.pad(centered, ((0, 0), (0, 3 - cols)))
    
    cov = np.cov(centered.T)
    
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]
        eigenvalues_sorted = eigenvalues[idx]
        projected = centered @ eigenvectors[:, :3]
        # Return variance explained too
        total_var = np.sum(eigenvalues)
        variance_explained = (eigenvalues_sorted[:3] / total_var * 100).tolist() if total_var > 0 else [0, 0, 0]
    except np.linalg.LinAlgError:
        projected = centered[:, :3]
        variance_explained = [0, 0, 0]
    
    return projected, variance_explained


def _find_nearest_neighbors(embedding_matrix: np.ndarray, token_idx: int, k: int = 10) -> list:
    """Find k nearest neighbors of a token by cosine similarity."""
    target = embedding_matrix[token_idx]
    target_norm = np.linalg.norm(target)
    if target_norm == 0:
        return []
    
    similarities = []
    for i in range(len(embedding_matrix)):
        if i == token_idx:
            continue
        norm_i = np.linalg.norm(embedding_matrix[i])
        if norm_i == 0:
            continue
        sim = float(np.dot(target, embedding_matrix[i]) / (target_norm * norm_i))
        similarities.append((i, sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:k]


def _compute_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ============== BPE Tokenizer ==============

class SimpleBPE:
    """
    Simple Byte-Pair Encoding tokenizer for educational purposes.
    Learns merges from the input text, then applies them.
    """
    
    def __init__(self, num_merges: int = 50):
        self.num_merges = num_merges
        self.merges = {}
        self.vocab = {}
    
    def _get_pairs(self, tokens: List[str]) -> Dict[tuple, int]:
        """Count all adjacent pairs."""
        pairs = {}
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pairs[pair] = pairs.get(pair, 0) + 1
        return pairs
    
    def fit_and_tokenize(self, text: str):
        """Learn BPE merges from text and tokenize it."""
        # Start with character-level tokens
        tokens = list(text)
        merge_history = []
        
        for i in range(self.num_merges):
            pairs = self._get_pairs(tokens)
            if not pairs:
                break
            
            # Find most frequent pair
            best_pair = max(pairs, key=pairs.get)
            
            if pairs[best_pair] < 2:
                break
            
            # Merge the best pair
            merged = best_pair[0] + best_pair[1]
            merge_history.append({
                "step": i + 1,
                "pair": [best_pair[0], best_pair[1]],
                "merged": merged,
                "frequency": pairs[best_pair]
            })
            
            # Apply merge
            new_tokens = []
            j = 0
            while j < len(tokens):
                if j < len(tokens) - 1 and tokens[j] == best_pair[0] and tokens[j + 1] == best_pair[1]:
                    new_tokens.append(merged)
                    j += 2
                else:
                    new_tokens.append(tokens[j])
                    j += 1
            tokens = new_tokens
        
        # Build vocabulary
        vocab = list(set(tokens))
        vocab.sort()
        token_to_id = {t: i for i, t in enumerate(vocab)}
        token_ids = [token_to_id[t] for t in tokens]
        
        return {
            "tokens": tokens,
            "token_ids": token_ids,
            "num_tokens": len(tokens),
            "vocabulary": vocab,
            "vocab_size": len(vocab),
            "merge_history": merge_history,
            "compression_ratio": round(len(text) / max(len(tokens), 1), 2)
        }


# ============== Endpoints ==============

@router.post("/attention")
async def compute_attention(request: AttentionComputeRequest):
    """
    Compute real attention weights from input text.
    Creates a temporary model, runs a forward pass, and returns attention data.
    """
    text = request.text
    tokens = text.split() if ' ' in text else list(text)
    
    if len(tokens) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 tokens. Use spaces to separate words or enter multiple characters.")
    if len(tokens) > 64:
        tokens = tokens[:64]
    
    seq_len = len(tokens)
    d_model = request.d_model
    num_heads = request.num_heads
    head_dim = d_model // num_heads
    
    if d_model % num_heads != 0:
        raise HTTPException(status_code=400, detail=f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
    
    np.random.seed(42)  # Deterministic for same input
    
    # Create token embeddings (deterministic from text)
    token_set = list(set(tokens))
    token_set.sort()
    token_to_id = {t: i for i, t in enumerate(token_set)}
    
    # Initialize random but deterministic embeddings
    embeddings = np.random.randn(len(token_set), d_model).astype(np.float32) * 0.1
    
    # Get embeddings for the input sequence
    input_emb = np.array([embeddings[token_to_id[t]] for t in tokens])
    
    # Add simple positional encoding
    pos_enc = np.zeros((seq_len, d_model), dtype=np.float32)
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            pos_enc[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
            if i + 1 < d_model:
                pos_enc[pos, i + 1] = np.cos(pos / (10000 ** (i / d_model)))
    input_emb = input_emb + pos_enc
    
    # Compute multi-head attention for each layer
    all_layers = []
    x = input_emb
    
    for layer_idx in range(request.num_layers):
        layer_seed = 42 + layer_idx * 100
        np.random.seed(layer_seed)
        
        # Initialize Q, K, V projection matrices
        W_q = np.random.randn(d_model, d_model).astype(np.float32) * (1.0 / np.sqrt(d_model))
        W_k = np.random.randn(d_model, d_model).astype(np.float32) * (1.0 / np.sqrt(d_model))
        W_v = np.random.randn(d_model, d_model).astype(np.float32) * (1.0 / np.sqrt(d_model))
        
        # Project Q, K, V
        Q = x @ W_q  # (seq_len, d_model)
        K = x @ W_k
        V = x @ W_v
        
        # Reshape for multi-head: (num_heads, seq_len, head_dim)
        Q_heads = Q.reshape(seq_len, num_heads, head_dim).transpose(1, 0, 2)
        K_heads = K.reshape(seq_len, num_heads, head_dim).transpose(1, 0, 2)
        V_heads = V.reshape(seq_len, num_heads, head_dim).transpose(1, 0, 2)
        
        heads_data = []
        for h in range(num_heads):
            q_h = Q_heads[h]  # (seq_len, head_dim)
            k_h = K_heads[h]
            v_h = V_heads[h]
            
            # Attention scores: Q @ K^T / sqrt(d_k)
            scores = (q_h @ k_h.T) / np.sqrt(head_dim)
            raw_scores_unmasked = scores.copy()
            
            # Apply causal mask
            if request.show_causal_mask:
                mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9
                scores = scores + mask
            
            # Softmax
            scores_exp = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
            attention_weights = scores_exp / (np.sum(scores_exp, axis=-1, keepdims=True) + 1e-10)
            
            # Output
            output = attention_weights @ v_h
            
            # === Analytics ===
            # Entropy per query token (higher = more uniform attention)
            eps = 1e-10
            entropy_per_token = (-np.sum(
                attention_weights * np.log(attention_weights + eps), axis=-1
            )).tolist()
            avg_entropy = float(np.mean(entropy_per_token))
            max_possible_entropy = float(np.log(seq_len))
            
            # Sparsity: fraction of weights near zero (<0.05)
            sparsity = float(np.mean(attention_weights < 0.05))
            
            # Dominant attention: for each query, which key gets the most weight
            dominant_indices = np.argmax(attention_weights, axis=-1).tolist()
            dominant_weights = np.max(attention_weights, axis=-1).tolist()
            
            # Per-token: how much total attention each token RECEIVES
            attention_received = np.sum(attention_weights, axis=0).tolist()
            
            # QK alignment (dot product norms for each pair)
            qk_norms = {
                "q_norms": np.linalg.norm(q_h, axis=-1).tolist(),
                "k_norms": np.linalg.norm(k_h, axis=-1).tolist(),
                "v_norms": np.linalg.norm(v_h, axis=-1).tolist(),
            }
            
            heads_data.append({
                "head": h,
                "attention_matrix": attention_weights.tolist(),
                "q_vectors": q_h[:, :min(8, head_dim)].tolist(),
                "k_vectors": k_h[:, :min(8, head_dim)].tolist(),
                "v_vectors": v_h[:, :min(8, head_dim)].tolist(),
                "raw_scores": raw_scores_unmasked.tolist(),
                # Analytics
                "entropy_per_token": entropy_per_token,
                "avg_entropy": avg_entropy,
                "max_entropy": max_possible_entropy,
                "sparsity": sparsity,
                "dominant_indices": dominant_indices,
                "dominant_weights": dominant_weights,
                "attention_received": attention_received,
                "qk_norms": qk_norms,
            })
        
        all_layers.append({
            "layer": layer_idx,
            "heads": heads_data
        })
        
        # Use the output as input to next layer (simplified)
        # Concatenate heads and project
        concat = np.concatenate([V_heads[h] for h in range(num_heads)], axis=-1)
        W_o = np.random.randn(d_model, d_model).astype(np.float32) * (1.0 / np.sqrt(d_model))
        x = x + concat @ W_o  # Residual connection
    
    return {
        "tokens": tokens,
        "seq_len": seq_len,
        "d_model": d_model,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "num_layers": request.num_layers,
        "layers": all_layers,
        "causal_mask": request.show_causal_mask,
        "token_embeddings": input_emb[:, :8].tolist(),
        "positional_encoding": pos_enc[:, :8].tolist(),
    }


@router.post("/embeddings")
async def compute_embeddings(request: EmbeddingComputeRequest):
    """
    Generate real embedding vectors with 2D and 3D projections using PCA.
    If text is provided, tokenize it and show its embeddings.
    Otherwise, show the full vocabulary embeddings.
    """
    np.random.seed(request.seed)
    
    vocab_size = request.vocab_size
    embedding_dim = request.embedding_dim
    
    # Create embedding matrix (deterministic from seed)
    embedding_matrix = np.random.randn(vocab_size, embedding_dim).astype(np.float32) * 0.1
    
    # Add some structure: make semantically "close" tokens cluster
    for i in range(vocab_size):
        if i < 128:
            c = chr(i)
            if c.isalpha():
                embedding_matrix[i, 0] += 0.5
                if c.isupper():
                    embedding_matrix[i, 1] += 0.3
                else:
                    embedding_matrix[i, 1] -= 0.3
                if c.lower() in 'aeiou':
                    embedding_matrix[i, 2] += 0.4
            elif c.isdigit():
                embedding_matrix[i, 0] -= 0.5
                embedding_matrix[i, 3] += 0.3 + int(c) * 0.05
            elif c in '.,;:!?':
                embedding_matrix[i, 4] += 0.5
    
    # Compute norms for every token
    norms = np.linalg.norm(embedding_matrix, axis=1).tolist()
    
    if request.text:
        text_tokens = list(request.text)
        text_ids = [ord(c) % vocab_size for c in text_tokens]
        text_embeddings = embedding_matrix[text_ids]
        
        # 2D PCA
        if len(text_tokens) > 1:
            text_projected_2d = _simple_pca_2d(text_embeddings)
        else:
            text_projected_2d = text_embeddings[:, :2]
        
        # 3D PCA
        if len(text_tokens) > 2 and embedding_dim >= 3:
            text_projected_3d, variance_explained = _simple_pca_3d(text_embeddings)
        else:
            text_projected_3d = np.pad(text_embeddings[:, :3], ((0, 0), (0, max(0, 3 - text_embeddings.shape[1]))))
            variance_explained = [0, 0, 0]
        
        # Pairwise similarities
        similarities = []
        for i in range(min(len(text_tokens), 20)):
            for j in range(i + 1, min(len(text_tokens), 20)):
                sim = _compute_cosine_similarity(text_embeddings[i], text_embeddings[j])
                similarities.append({
                    "token_a": text_tokens[i],
                    "token_b": text_tokens[j],
                    "idx_a": i,
                    "idx_b": j,
                    "similarity": round(sim, 4)
                })
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        
        return {
            "mode": "text",
            "text": request.text,
            "tokens": text_tokens,
            "token_ids": text_ids,
            "projections": text_projected_2d.tolist(),
            "projections_3d": text_projected_3d.tolist(),
            "variance_explained": variance_explained,
            "embedding_dim": embedding_dim,
            "vocab_size": vocab_size,
            "norms": [float(np.linalg.norm(text_embeddings[i])) for i in range(len(text_tokens))],
            "similarities": similarities[:20],
            "sample_vectors": [
                {"token": text_tokens[i], "vector": text_embeddings[i].tolist()}
                for i in range(min(5, len(text_tokens)))
            ]
        }
    else:
        # Full vocabulary projection
        projected_2d = _simple_pca_2d(embedding_matrix)
        projected_3d_result = _simple_pca_3d(embedding_matrix)
        projected_3d = projected_3d_result[0]
        variance_explained = projected_3d_result[1]
        
        token_labels = []
        for i in range(vocab_size):
            if i < 128 and chr(i).isprintable() and not chr(i).isspace():
                token_labels.append(chr(i))
            elif i < 128 and chr(i) == ' ':
                token_labels.append('\u2423')
            else:
                token_labels.append(f't{i}')
        
        # Compute top similarity pairs (for similarity lines)
        top_pairs = []
        sample_count = min(vocab_size, 80)  # Only compute for printable ASCII
        for i in range(sample_count):
            for j in range(i + 1, sample_count):
                sim = _compute_cosine_similarity(embedding_matrix[i], embedding_matrix[j])
                if sim > 0.7:  # Only high similarity pairs
                    top_pairs.append({
                        "idx_a": i, "idx_b": j,
                        "token_a": token_labels[i], "token_b": token_labels[j],
                        "similarity": round(sim, 4)
                    })
        top_pairs.sort(key=lambda x: x["similarity"], reverse=True)
        
        return {
            "mode": "vocabulary",
            "vocab_size": vocab_size,
            "embedding_dim": embedding_dim,
            "projections": projected_2d.tolist(),
            "projections_3d": projected_3d.tolist(),
            "variance_explained": variance_explained,
            "tokens": token_labels,
            "norms": norms,
            "top_similarity_pairs": top_pairs[:50],
            "sample_vectors": [
                {"token": token_labels[i], "vector": embedding_matrix[i].tolist()}
                for i in range(min(5, vocab_size))
            ],
            "embedding_stats": {
                "mean": float(np.mean(embedding_matrix)),
                "std": float(np.std(embedding_matrix)),
                "min": float(np.min(embedding_matrix)),
                "max": float(np.max(embedding_matrix)),
            }
        }


@router.post("/embeddings/similarity")
async def compute_embedding_similarity(
    token_a: str,
    token_b: str,
    embedding_dim: int = 64,
    seed: int = 42
):
    """Compute cosine similarity between two token embeddings."""
    np.random.seed(seed)
    emb = np.random.randn(256, embedding_dim).astype(np.float32) * 0.1
    
    id_a = ord(token_a[0]) % 256 if token_a else 0
    id_b = ord(token_b[0]) % 256 if token_b else 0
    
    vec_a = emb[id_a]
    vec_b = emb[id_b]
    
    sim = _compute_cosine_similarity(vec_a, vec_b)
    
    return {
        "token_a": token_a,
        "token_b": token_b,
        "similarity": round(sim, 6),
        "vector_a": vec_a[:8].tolist(),
        "vector_b": vec_b[:8].tolist(),
        "dot_product": float(np.dot(vec_a, vec_b)),
        "norm_a": float(np.linalg.norm(vec_a)),
        "norm_b": float(np.linalg.norm(vec_b)),
    }


@router.post("/embeddings/arithmetic")
async def compute_embedding_arithmetic(
    token_a: str,
    token_b: str,
    token_c: str,
    embedding_dim: int = 64,
    seed: int = 42,
    vocab_size: int = 128
):
    """
    Compute embedding arithmetic: A - B + C = ?
    Returns the closest token in the vocabulary.
    """
    np.random.seed(seed)
    emb = np.random.randn(vocab_size, embedding_dim).astype(np.float32) * 0.1
    
    id_a = ord(token_a[0]) % vocab_size if token_a else 0
    id_b = ord(token_b[0]) % vocab_size if token_b else 0
    id_c = ord(token_c[0]) % vocab_size if token_c else 0
    
    vec_a = emb[id_a]
    vec_b = emb[id_b]
    vec_c = emb[id_c]
    
    # Compute A - B + C
    result_vec = vec_a - vec_b + vec_c
    
    # Find closest token
    similarities = []
    for i in range(vocab_size):
        sim = _compute_cosine_similarity(result_vec, emb[i])
        label = chr(i) if i < 128 and chr(i).isprintable() else f't{i}'
        similarities.append({"id": i, "token": label, "similarity": round(sim, 4)})
    
    similarities.sort(key=lambda x: x["similarity"], reverse=True)
    
    return {
        "expression": f"{token_a} - {token_b} + {token_c}",
        "result_vector": result_vec[:8].tolist(),
        "top_matches": similarities[:10],
    }


@router.post("/sampling")
async def compute_sampling(request: SamplingRequest):
    """
    Demonstrate how different sampling strategies affect token selection.
    Given logits (or generates random ones), shows the probability
    distribution under different temperature/top-k/top-p settings.
    """
    vocab_size = request.vocab_size
    
    if request.logits and len(request.logits) > 0:
        logits = np.array(request.logits, dtype=np.float32)
        vocab_size = len(logits)
    else:
        # Generate realistic logits (most tokens low, few high)
        np.random.seed(hash(request.text or "demo") % 2**31)
        logits = np.random.randn(vocab_size).astype(np.float32) * 2.0
        # Make some tokens more likely
        top_indices = np.random.choice(vocab_size, size=min(10, vocab_size), replace=False)
        logits[top_indices] += np.random.uniform(2.0, 5.0, size=len(top_indices))
    
    # Apply temperature
    scaled_logits = logits / max(request.temperature, 0.01)
    
    # Softmax
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
    probs = exp_logits / np.sum(exp_logits)
    
    # Sort by probability
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    
    # Top-k filtering
    top_k = min(request.top_k, vocab_size)
    top_k_indices = sorted_indices[:top_k]
    top_k_probs = sorted_probs[:top_k]
    top_k_probs = top_k_probs / np.sum(top_k_probs)  # Renormalize
    
    # Top-p (nucleus) filtering
    cumulative = np.cumsum(sorted_probs)
    nucleus_mask = cumulative <= request.top_p
    nucleus_mask[0] = True  # Always include the top token
    nucleus_indices = sorted_indices[nucleus_mask]
    nucleus_probs = sorted_probs[nucleus_mask]
    nucleus_probs = nucleus_probs / np.sum(nucleus_probs)
    
    # Token labels
    def get_label(idx):
        if idx < 128 and chr(idx).isprintable():
            return chr(idx)
        return f"<t{idx}>"
    
    return {
        "temperature": request.temperature,
        "top_k": request.top_k,
        "top_p": request.top_p,
        "vocab_size": vocab_size,
        "original_distribution": [
            {"token_id": int(sorted_indices[i]), "token": get_label(int(sorted_indices[i])), 
             "probability": float(sorted_probs[i]), "logit": float(logits[sorted_indices[i]])}
            for i in range(min(30, vocab_size))
        ],
        "top_k_distribution": [
            {"token_id": int(top_k_indices[i]), "token": get_label(int(top_k_indices[i])),
             "probability": float(top_k_probs[i])}
            for i in range(min(20, len(top_k_indices)))
        ],
        "nucleus_distribution": [
            {"token_id": int(nucleus_indices[i]), "token": get_label(int(nucleus_indices[i])),
             "probability": float(nucleus_probs[i])}
            for i in range(min(20, len(nucleus_indices)))
        ],
        "entropy": float(-np.sum(probs * np.log(probs + 1e-10))),
        "top_1_prob": float(sorted_probs[0]),
        "top_5_cumulative": float(np.sum(sorted_probs[:5])),
    }


@router.post("/forward-step")
async def compute_forward_step(request: ForwardStepRequest):
    """
    Step-by-step transformer forward pass visualization.
    Returns intermediate values for each step of the computation.
    """
    text = request.text
    tokens = text.split() if ' ' in text else list(text)
    if len(tokens) > 32:
        tokens = tokens[:32]
    
    seq_len = len(tokens)
    d_model = request.d_model
    num_heads = request.num_heads
    head_dim = d_model // num_heads
    
    np.random.seed(42)
    
    # Token embeddings
    token_set = list(set(tokens))
    token_set.sort()
    token_to_id = {t: i for i, t in enumerate(token_set)}
    emb_matrix = np.random.randn(len(token_set), d_model).astype(np.float32) * 0.1
    
    input_emb = np.array([emb_matrix[token_to_id[t]] for t in tokens])
    
    # Positional encoding
    pos_enc = np.zeros((seq_len, d_model), dtype=np.float32)
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            pos_enc[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
            if i + 1 < d_model:
                pos_enc[pos, i + 1] = np.cos(pos / (10000 ** (i / d_model)))
    
    x = input_emb + pos_enc
    
    step = request.step
    result = {
        "step": step,
        "tokens": tokens,
        "shape": f"({seq_len}, {d_model})",
    }
    
    if step == 0:
        result["name"] = "Token + Position Embeddings"
        result["description"] = "Convert tokens to vectors and add positional information"
        result["data"] = {
            "token_embeddings": input_emb[:, :8].tolist(),
            "positional_encoding": pos_enc[:, :8].tolist(),
            "combined": x[:, :8].tolist(),
            "embedding_norms": np.linalg.norm(input_emb, axis=1).tolist(),
        }
    elif step == 1:
        # RMSNorm
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + 1e-6)
        normed = x / rms
        result["name"] = "RMSNorm (Pre-Attention)"
        result["description"] = "Normalize input before the attention layer"
        result["data"] = {
            "input": x[:, :8].tolist(),
            "rms_values": rms.flatten().tolist(),
            "normalized": normed[:, :8].tolist(),
            "formula": "x / √(mean(x²) + ε)",
        }
    elif step == 2:
        # Q, K, V projections
        W_q = np.random.randn(d_model, d_model).astype(np.float32) * (1.0 / np.sqrt(d_model))
        W_k = np.random.randn(d_model, d_model).astype(np.float32) * (1.0 / np.sqrt(d_model))
        W_v = np.random.randn(d_model, d_model).astype(np.float32) * (1.0 / np.sqrt(d_model))
        Q = x @ W_q
        K = x @ W_k
        V = x @ W_v
        result["name"] = "Q, K, V Projections"
        result["description"] = "Project input into Query, Key, and Value spaces"
        result["data"] = {
            "Q": Q[:, :8].tolist(),
            "K": K[:, :8].tolist(),
            "V": V[:, :8].tolist(),
            "Q_norm": np.linalg.norm(Q, axis=1).tolist(),
            "K_norm": np.linalg.norm(K, axis=1).tolist(),
            "V_norm": np.linalg.norm(V, axis=1).tolist(),
            "formula": "Q = X·W_q, K = X·W_k, V = X·W_v",
        }
    elif step == 3:
        # Attention scores
        W_q = np.random.randn(d_model, d_model).astype(np.float32) * (1.0 / np.sqrt(d_model))
        W_k = np.random.randn(d_model, d_model).astype(np.float32) * (1.0 / np.sqrt(d_model))
        Q = x @ W_q
        K = x @ W_k
        scores = (Q @ K.T) / np.sqrt(d_model)
        mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9
        masked_scores = scores + mask
        result["name"] = "Attention Scores + Causal Mask"
        result["description"] = "Compute compatibility scores and apply causal mask"
        result["data"] = {
            "raw_scores": scores.tolist(),
            "masked_scores": masked_scores.tolist(),
            "formula": "Scores = Q·K^T / √d_k, then mask future positions",
        }
    elif step == 4:
        # Softmax
        W_q = np.random.randn(d_model, d_model).astype(np.float32) * (1.0 / np.sqrt(d_model))
        W_k = np.random.randn(d_model, d_model).astype(np.float32) * (1.0 / np.sqrt(d_model))
        Q = x @ W_q
        K = x @ W_k
        scores = (Q @ K.T) / np.sqrt(d_model)
        mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9
        scores = scores + mask
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        weights = exp_scores / (np.sum(exp_scores, axis=-1, keepdims=True) + 1e-10)
        result["name"] = "Softmax (Attention Weights)"
        result["description"] = "Normalize scores to probabilities"
        result["data"] = {
            "attention_weights": weights.tolist(),
            "row_sums": np.sum(weights, axis=1).tolist(),
            "formula": "Weights = softmax(Scores)",
        }
    elif step == 5:
        # MLP
        result["name"] = "Feedforward Network (MLP)"
        result["description"] = "Position-wise transformation with non-linearity"
        d_ff = d_model * 4
        W1 = np.random.randn(d_model, d_ff).astype(np.float32) * (1.0 / np.sqrt(d_model))
        hidden = x @ W1
        # GELU approximation
        activated = hidden * 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (hidden + 0.044715 * hidden ** 3)))
        W2 = np.random.randn(d_ff, d_model).astype(np.float32) * (1.0 / np.sqrt(d_ff))
        output = activated @ W2
        result["data"] = {
            "hidden_pre_activation": hidden[:, :8].tolist(),
            "hidden_post_activation": activated[:, :8].tolist(),
            "output": output[:, :8].tolist(),
            "expansion_ratio": d_ff / d_model,
            "formula": "MLP(x) = GELU(x·W₁)·W₂",
        }
    elif step == 6:
        # Residual
        residual = x + np.random.randn(*x.shape).astype(np.float32) * 0.01
        result["name"] = "Residual Connection"
        result["description"] = "Add the input to the output (skip connection)"
        result["data"] = {
            "input_norm": float(np.linalg.norm(x)),
            "output_norm": float(np.linalg.norm(residual)),
            "difference_norm": float(np.linalg.norm(residual - x)),
            "formula": "output = x + sublayer(x)",
        }
    else:
        # Final output logits
        vocab_size = max(len(token_set), 32)
        W_out = np.random.randn(d_model, vocab_size).astype(np.float32) * (1.0 / np.sqrt(d_model))
        logits = x @ W_out
        probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = probs / np.sum(probs, axis=-1, keepdims=True)
        
        last_probs = probs[-1]
        top_ids = np.argsort(last_probs)[::-1][:10]
        
        result["name"] = "Output Logits & Predictions"
        result["description"] = "Project to vocabulary and get token probabilities"
        result["data"] = {
            "logits_shape": f"({seq_len}, {vocab_size})",
            "top_predictions": [
                {"token_id": int(idx), "probability": float(last_probs[idx])}
                for idx in top_ids
            ],
            "formula": "logits = x · W_out, probs = softmax(logits)",
        }
    
    return result


# ============== Dataset Endpoints ==============

# Built-in sample datasets
BUILTIN_DATASETS = {
    "shakespeare": {
        "name": "Shakespeare",
        "description": "Excerpt from Shakespeare's works",
        "size": "~2K chars",
        "text": """To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles,
And by opposing end them. To die, to sleep—
No more—and by a sleep to say we end
The heartache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wished. To die, to sleep—
To sleep, perchance to dream—ay, there's the rub,
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause. There's the respect
That makes calamity of so long life.
For who would bear the whips and scorns of time,
The oppressor's wrong, the proud man's contumely,
The pangs of despised love, the law's delay,
The insolence of office, and the spurns
That patient merit of the unworthy takes,
When he himself might his quietus make
With a bare bodkin? Who would fardels bear,
To grunt and sweat under a weary life,
But that the dread of something after death,
The undiscovered country from whose bourn
No traveller returns, puzzles the will
And makes us rather bear those ills we have
Than fly to others that we know not of?
Thus conscience does make cowards of us all,
And thus the native hue of resolution
Is sicklied o'er with the pale cast of thought."""
    },
    "alice": {
        "name": "Alice in Wonderland",
        "description": "Opening of Alice's Adventures in Wonderland",
        "size": "~1.5K chars",
        "text": """Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, and what is the use of a book, thought Alice without pictures or conversation?

So she was considering in her own mind (as well as she could, for the hot day made her feel very sleepy and stupid), whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit with pink eyes ran close by her.

There was nothing so very remarkable in that; nor did Alice think it so very much out of the way to hear the Rabbit say to itself, Oh dear! Oh dear! I shall be late! (when she thought it over afterwards, it occurred to her that she ought to have wondered at this, but at the time it all seemed quite natural); but when the Rabbit actually took a watch out of its waistcoat-pocket, and looked at it, and then hurried on, Alice started to her feet."""
    },
    "code": {
        "name": "Python Code",
        "description": "Simple Python programming examples",
        "size": "~1K chars",
        "text": """def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

class LinkedList:
    def __init__(self, value=None):
        self.value = value
        self.next = None
    
    def append(self, value):
        if self.value is None:
            self.value = value
            return
        current = self
        while current.next:
            current = current.next
        current.next = LinkedList(value)"""
    }
}

# User-uploaded datasets storage (in-memory for demo)
_custom_datasets: Dict[str, Dict[str, Any]] = {}


@router.get("/datasets")
async def list_datasets():
    """List all available datasets (built-in + uploaded)."""
    datasets = []
    
    for key, ds in BUILTIN_DATASETS.items():
        datasets.append({
            "id": key,
            "name": ds["name"],
            "description": ds["description"],
            "size": ds["size"],
            "type": "builtin",
            "char_count": len(ds["text"]),
            "preview": ds["text"][:200] + "..."
        })
    
    for key, ds in _custom_datasets.items():
        datasets.append({
            "id": key,
            "name": ds["name"],
            "description": "User-uploaded dataset",
            "size": f"~{len(ds['text'])} chars",
            "type": "custom",
            "char_count": len(ds["text"]),
            "preview": ds["text"][:200] + "..."
        })
    
    return {"datasets": datasets}


@router.get("/datasets/{dataset_id}")
async def get_dataset(dataset_id: str):
    """Get a specific dataset's text content."""
    if dataset_id in BUILTIN_DATASETS:
        ds = BUILTIN_DATASETS[dataset_id]
        text = ds["text"]
    elif dataset_id in _custom_datasets:
        text = _custom_datasets[dataset_id]["text"]
    else:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Character-level stats
    chars = list(text)
    unique_chars = list(set(chars))
    unique_chars.sort()
    char_freq = {}
    for c in chars:
        display = c if c.isprintable() and c != ' ' else ('␣' if c == ' ' else repr(c))
        char_freq[display] = char_freq.get(display, 0) + 1
    
    return {
        "id": dataset_id,
        "text": text,
        "stats": {
            "total_chars": len(text),
            "unique_chars": len(unique_chars),
            "total_words": len(text.split()),
            "total_lines": text.count('\n') + 1,
            "char_frequencies": dict(sorted(char_freq.items(), key=lambda x: -x[1])[:30])
        }
    }


@router.post("/datasets/upload")
async def upload_dataset(name: str = Form(...), text: str = Form(...)):
    """Upload a custom text dataset."""
    if len(text) < 10:
        raise HTTPException(status_code=400, detail="Text must be at least 10 characters")
    if len(text) > 100000:
        raise HTTPException(status_code=400, detail="Text must be under 100,000 characters")
    
    import uuid
    dataset_id = f"custom_{str(uuid.uuid4())[:8]}"
    
    _custom_datasets[dataset_id] = {
        "name": name,
        "text": text
    }
    
    return {
        "id": dataset_id,
        "name": name,
        "char_count": len(text),
        "status": "uploaded"
    }


# ============== Improved BPE Tokenization ==============

@router.post("/tokenize/bpe")
async def tokenize_bpe(text: str, num_merges: int = 50):
    """
    Run real BPE tokenization with merge history for visualization.
    """
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    bpe = SimpleBPE(num_merges=min(num_merges, 200))
    result = bpe.fit_and_tokenize(text)
    result["strategy"] = "bpe"
    result["text"] = text
    
    return result
