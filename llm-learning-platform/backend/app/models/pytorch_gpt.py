"""
PyTorch-native GPT model with GPU acceleration.
Mirrors the custom MicroGPT architecture but uses torch.nn for real performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List


# =============================================================================
# DEVICE SELECTION
# =============================================================================

def get_device() -> torch.device:
    """Auto-detect the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = get_device()


# =============================================================================
# MODEL CONFIG
# =============================================================================

@dataclass
class PyTorchGPTConfig:
    """Configuration for PyTorch GPT model."""
    vocab_size: int = 256          # Character-level by default
    max_seq_len: int = 256
    d_model: int = 128
    num_layers: int = 4
    num_heads: int = 4
    d_ff: int = 512               # Feedforward dimension
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation: str = "gelu"       # gelu, relu, swiglu
    norm_type: str = "rmsnorm"     # rmsnorm, layernorm
    tie_weights: bool = True
    use_bias: bool = False
    use_flash_attention: bool = True  # Use PyTorch SDPA (FlashAttention-2)

    def __post_init__(self):
        if self.d_ff == 0:
            self.d_ff = 4 * self.d_model


# =============================================================================
# LAYERS
# =============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.
    Used in modern LLMs (Llama, Mistral, etc.) for better training stability.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention with optional FlashAttention.

    Supports:
    - Standard scaled dot-product attention
    - PyTorch SDPA (FlashAttention-2 when available)
    - Causal masking for autoregressive generation
    - KV-cache for fast inference
    """
    def __init__(self, config: PyTorchGPTConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.d_model // config.num_heads
        assert config.d_model % config.num_heads == 0, "d_model must be divisible by num_heads"

        # Q, K, V projections
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=config.use_bias)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=config.use_bias)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=config.use_bias)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.use_bias)

        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.use_flash = config.use_flash_attention

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = x.shape

        # Project Q, K, V
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # KV-cache for inference
        new_kv_cache = None
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
            new_kv_cache = (k, v)

        attn_weights = None

        # Use PyTorch SDPA (FlashAttention-2 when hardware supports it)
        if self.use_flash and not return_attention:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=self.config.attention_dropout if self.training else 0.0,
                is_causal=(attention_mask is None and kv_cache is None),
            )
        else:
            # Manual attention for visualization
            scale = 1.0 / math.sqrt(self.head_dim)
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

            # Causal mask
            if attention_mask is None and kv_cache is None:
                causal_mask = torch.triu(
                    torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
                )
                attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))
            elif attention_mask is not None:
                attn_weights = attn_weights + attention_mask

            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            y = torch.matmul(attn_weights, v)

        # Merge heads and project
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.out_proj(y))

        return y, attn_weights, new_kv_cache


class MLP(nn.Module):
    """Feedforward network with configurable activation.
    Supports GELU, ReLU, and SwiGLU.
    """
    def __init__(self, config: PyTorchGPTConfig):
        super().__init__()
        self.config = config

        if config.activation == "swiglu":
            # SwiGLU: split into gate and value
            self.w1 = nn.Linear(config.d_model, config.d_ff, bias=config.use_bias)
            self.w2 = nn.Linear(config.d_ff, config.d_model, bias=config.use_bias)
            self.w3 = nn.Linear(config.d_model, config.d_ff, bias=config.use_bias)
        else:
            self.fc1 = nn.Linear(config.d_model, config.d_ff, bias=config.use_bias)
            self.fc2 = nn.Linear(config.d_ff, config.d_model, bias=config.use_bias)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.activation == "swiglu":
            return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
        elif self.config.activation == "gelu":
            return self.dropout(self.fc2(F.gelu(self.fc1(x))))
        else:  # relu
            return self.dropout(self.fc2(F.relu(self.fc1(x))))


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block: Norm -> Attention -> Residual -> Norm -> MLP -> Residual"""
    def __init__(self, config: PyTorchGPTConfig, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx

        # Pre-norm
        if config.norm_type == "rmsnorm":
            self.norm1 = RMSNorm(config.d_model)
            self.norm2 = RMSNorm(config.d_model)
        else:
            self.norm1 = nn.LayerNorm(config.d_model)
            self.norm2 = nn.LayerNorm(config.d_model)

        self.attn = MultiHeadAttention(config)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Attention with residual
        normed = self.norm1(x)
        attn_out, attn_weights, new_kv_cache = self.attn(
            normed, attention_mask, kv_cache, return_attention
        )
        x = x + attn_out

        # MLP with residual
        x = x + self.mlp(self.norm2(x))

        return x, attn_weights, new_kv_cache


# =============================================================================
# FULL GPT MODEL
# =============================================================================

class PyTorchGPT(nn.Module):
    """PyTorch-native GPT model with GPU support.

    Architecture follows GPT-2/GPT-3 with modern improvements:
    - RMSNorm (default) or LayerNorm
    - SwiGLU / GELU / ReLU activations
    - Rotary positional embeddings ready (uses learned for now)
    - FlashAttention-2 via PyTorch SDPA
    - KV-cache for fast autoregressive generation
    - Mixed precision support
    """
    def __init__(self, config: PyTorchGPTConfig):
        super().__init__()
        self.config = config

        # Token and position embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config, layer_idx=i)
            for i in range(config.num_layers)
        ])

        # Final norm and output projection
        if config.norm_type == "rmsnorm":
            self.final_norm = RMSNorm(config.d_model)
        else:
            self.final_norm = nn.LayerNorm(config.d_model)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        if config.tie_weights:
            self.lm_head.weight = self.token_embedding.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Report parameter count
        self.num_parameters = sum(p.numel() for p in self.parameters())

    def _init_weights(self, module: nn.Module):
        """Initialize weights with scaled normal distribution."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        return_attention: bool = False,
    ) -> Dict[str, Any]:
        """
        Forward pass.

        Args:
            input_ids: (batch_size, seq_len) token IDs
            attention_mask: Optional mask
            kv_cache: Optional KV-cache list (one per layer)
            return_attention: Whether to return attention weights

        Returns:
            Dict with 'logits', optionally 'attention_weights' and 'kv_cache'
        """
        B, T = input_ids.shape
        device = input_ids.device

        # Embed tokens + positions
        pos_offset = kv_cache[0][0].shape[2] if kv_cache is not None else 0
        positions = torch.arange(pos_offset, pos_offset + T, device=device)
        tok_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(positions)
        x = self.embedding_dropout(tok_emb + pos_emb)

        # Run through transformer blocks
        all_attn_weights = []
        new_kv_cache = []

        for i, block in enumerate(self.blocks):
            layer_kv = kv_cache[i] if kv_cache is not None else None
            x, attn_weights, new_layer_kv = block(
                x, attention_mask, layer_kv, return_attention
            )
            if return_attention and attn_weights is not None:
                all_attn_weights.append(attn_weights)
            if new_layer_kv is not None:
                new_kv_cache.append(new_layer_kv)

        # Final norm + projection
        x = self.final_norm(x)
        logits = self.lm_head(x)

        result = {"logits": logits}
        if return_attention:
            result["attention_weights"] = all_attn_weights
        if new_kv_cache:
            result["kv_cache"] = new_kv_cache

        return result

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = 0.9,
        repetition_penalty: float = 1.0,
        use_kv_cache: bool = True,
    ) -> torch.Tensor:
        """
        Autoregressive text generation with KV-cache support.

        Args:
            input_ids: (1, seq_len) prompt token IDs
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-K filtering
            top_p: Nucleus (top-p) filtering
            repetition_penalty: Penalty for repeated tokens
            use_kv_cache: Whether to use KV-cache for speed

        Returns:
            (1, seq_len + max_new_tokens) generated token IDs
        """
        self.eval()
        kv_cache = None
        generated = input_ids

        for _ in range(max_new_tokens):
            # Truncate to max_seq_len if needed
            if use_kv_cache and kv_cache is not None:
                # Only feed the last token
                curr_input = generated[:, -1:]
            else:
                curr_input = generated[:, -self.config.max_seq_len:]
                kv_cache = None

            result = self.forward(curr_input, kv_cache=kv_cache if use_kv_cache else None)
            logits = result["logits"][:, -1, :]  # (B, vocab_size)

            if use_kv_cache and "kv_cache" in result:
                kv_cache = result["kv_cache"]

            # Repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(generated[0].tolist()):
                    logits[0, token_id] /= repetition_penalty

            # Temperature scaling
            logits = logits / temperature

            # Top-K filtering
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, -1:]] = float("-inf")

            # Top-P (nucleus) filtering
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # Remove tokens above the threshold
                sorted_mask = cum_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[sorted_mask] = float("-inf")
                # Scatter back
                logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

        return generated

    def get_num_parameters(self, non_embedding: bool = False) -> int:
        """Count parameters, optionally excluding embeddings."""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.token_embedding.weight.numel()
            n -= self.position_embedding.weight.numel()
        return n

    def estimate_memory_mb(self) -> float:
        """Estimate GPU memory usage in MB."""
        param_bytes = sum(p.numel() * p.element_size() for p in self.parameters())
        # Rough estimate: params + gradients + optimizer states (Adam ~3x)
        total = param_bytes * 4  # params + grads + 2x optimizer
        return total / (1024 * 1024)

    def get_layer_info(self) -> List[Dict[str, Any]]:
        """Get information about each layer for visualization."""
        layers = []
        for name, param in self.named_parameters():
            layers.append({
                "name": name,
                "shape": list(param.shape),
                "num_params": param.numel(),
                "requires_grad": param.requires_grad,
                "dtype": str(param.dtype),
                "device": str(param.device),
                "grad_norm": param.grad.norm().item() if param.grad is not None else None,
            })
        return layers

    def get_weight_stats(self) -> Dict[str, Any]:
        """Get weight statistics for visualization."""
        all_weights = []
        layer_stats = {}
        for name, param in self.named_parameters():
            data = param.detach().cpu().flatten().numpy()
            all_weights.extend(data.tolist())
            layer_stats[name] = {
                "mean": float(param.mean()),
                "std": float(param.std()),
                "min": float(param.min()),
                "max": float(param.max()),
                "norm": float(param.norm()),
            }
        return {
            "total_params": self.get_num_parameters(),
            "layer_stats": layer_stats,
        }
