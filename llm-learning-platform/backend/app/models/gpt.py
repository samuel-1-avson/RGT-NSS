"""
GPT (Generative Pre-trained Transformer) Model Implementation
Built from scratch for educational purposes.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass

from app.core.tensor import Tensor, cross_entropy_loss
from app.core.module import Module, Linear, Embedding


@dataclass
class GPTConfig:
    """Configuration for GPT model."""
    vocab_size: int = 256
    max_seq_len: int = 256
    d_model: int = 128
    num_layers: int = 4
    num_heads: int = 4
    d_ff: int = 512
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation: str = 'gelu'
    norm_type: str = 'rmsnorm'  # 'rmsnorm' or 'layernorm'
    tie_weights: bool = True
    use_bias: bool = False  # Modern LLMs often don't use bias
    
    def __post_init__(self):
        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads"
        self.d_head = self.d_model // self.num_heads


class RMSNorm(Module):
    """
    Root Mean Square Layer Normalization.
    Used in modern LLMs like Llama for better training stability.
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Learnable scale parameter (gamma)
        self.weight = Tensor.ones(dim)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
        
        Args:
            x: Input tensor of shape (..., dim)
        Returns:
            Normalized tensor of same shape
        """
        # Calculate mean of squares
        mean_sq = np.mean(x.data ** 2, axis=-1, keepdims=True)
        
        # Normalize
        normalized = x.data / np.sqrt(mean_sq + self.eps)
        
        # Scale
        out_data = normalized * self.weight.data
        
        out = Tensor(out_data, (x, self.weight), 'rmsnorm', requires_grad=True)
        
        def _backward():
            if x.requires_grad:
                # Gradient w.r.t. input
                grad_normalized = out.grad * self.weight.data
                
                # Gradient through the division
                inv_rms = 1.0 / np.sqrt(mean_sq + self.eps)
                grad_input = grad_normalized * inv_rms
                
                # Gradient from mean constraint
                grad_mean_sq = -0.5 * np.mean(x.data * grad_normalized * (inv_rms ** 3), axis=-1, keepdims=True)
                grad_input += 2 * x.data * grad_mean_sq / x.shape[-1]
                
                x.grad += grad_input
            
            if self.weight.requires_grad:
                # Gradient w.r.t. weight
                self.weight.grad += np.sum(out.grad * normalized, axis=tuple(range(len(normalized.shape) - 1)))
        
        out._backward = _backward
        return out


class LayerNorm(Module):
    """Traditional Layer Normalization."""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = Tensor.ones(dim)
        self.bias = Tensor.zeros(dim)
    
    def forward(self, x: Tensor) -> Tensor:
        mean = np.mean(x.data, axis=-1, keepdims=True)
        var = np.var(x.data, axis=-1, keepdims=True)
        normalized = (x.data - mean) / np.sqrt(var + self.eps)
        out_data = normalized * self.weight.data + self.bias.data
        
        out = Tensor(out_data, (x, self.weight, self.bias), 'layernorm', requires_grad=True)
        
        def _backward():
            if x.requires_grad:
                grad_normalized = out.grad * self.weight.data
                grad_var = np.sum(grad_normalized * (x.data - mean) * -0.5 * (var + self.eps) ** (-1.5), axis=-1, keepdims=True)
                grad_mean = np.sum(grad_normalized * -1 / np.sqrt(var + self.eps), axis=-1, keepdims=True) + grad_var * np.sum(-2 * (x.data - mean), axis=-1, keepdims=True) / x.shape[-1]
                x.grad += grad_normalized / np.sqrt(var + self.eps) + grad_var * 2 * (x.data - mean) / x.shape[-1] + grad_mean / x.shape[-1]
            
            if self.weight.requires_grad:
                self.weight.grad += np.sum(out.grad * normalized, axis=tuple(range(len(normalized.shape) - 1)))
            
            if self.bias.requires_grad:
                self.bias.grad += np.sum(out.grad, axis=tuple(range(len(out.grad.shape) - 1)))
        
        out._backward = _backward
        return out


class MultiHeadAttention(Module):
    """
    Multi-Head Self-Attention mechanism.
    
    This is the core of the Transformer architecture.
    """
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.d_head = config.d_head
        
        # Q, K, V projections (combined for efficiency)
        self.q_proj = Linear(config.d_model, config.d_model, bias=config.use_bias)
        self.k_proj = Linear(config.d_model, config.d_model, bias=config.use_bias)
        self.v_proj = Linear(config.d_model, config.d_model, bias=config.use_bias)
        
        # Output projection
        self.o_proj = Linear(config.d_model, config.d_model, bias=config.use_bias)
        
        # Dropout
        self.attn_dropout = config.attention_dropout
        self.resid_dropout = config.dropout
        
        # Pre-compute causal mask
        self.register_buffer('causal_mask', self._create_causal_mask(config.max_seq_len))
    
    def _create_causal_mask(self, size: int) -> np.ndarray:
        """Create causal (autoregressive) mask."""
        mask = np.triu(np.ones((size, size)), k=1) * float('-inf')
        return mask
    
    def register_buffer(self, name: str, value: np.ndarray):
        """Register a persistent buffer."""
        self._buffers[name] = value
    
    def forward(self, x: Tensor, attention_mask: Optional[np.ndarray] = None, 
                store_attention: bool = False) -> Tuple[Tensor, Optional[np.ndarray]]:
        """
        Forward pass of multi-head attention.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            attention_mask: Optional mask (batch_size, seq_len)
            store_attention: Whether to return attention weights
        
        Returns:
            output: (batch_size, seq_len, d_model)
            attention_weights: Optional (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        Q = self.q_proj(x)  # (batch, seq, d_model)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Reshape for multi-head attention
        # (batch, seq, d_model) -> (batch, seq, num_heads, d_head) -> (batch, num_heads, seq, d_head)
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.d_head).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.d_head).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.d_head).transpose(0, 2, 1, 3)
        
        # Compute attention scores: Q @ K^T / sqrt(d_head)
        scores = (Q @ K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_head)
        
        # Apply causal mask
        causal_mask = self._buffers['causal_mask'][:seq_len, :seq_len]
        scores = scores + causal_mask  # Broadcasting
        
        # Apply optional padding mask
        if attention_mask is not None:
            # attention_mask: (batch, seq) -> (batch, 1, 1, seq)
            mask = attention_mask[:, None, None, :] * float('-inf')
            scores = scores + mask
        
        # Softmax
        attn_weights = scores.softmax(dim=-1)
        
        # Apply dropout to attention weights
        if self.training and self.attn_dropout > 0:
            attn_weights = attn_weights.dropout(self.attn_dropout, training=True)
        
        # Apply attention to values
        out = attn_weights @ V  # (batch, num_heads, seq, d_head)
        
        # Reshape back
        out = out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        
        # Output projection
        out = self.o_proj(out)
        
        # Residual dropout
        if self.training and self.resid_dropout > 0:
            out = out.dropout(self.resid_dropout, training=True)
        
        if store_attention:
            return out, attn_weights.data
        return out, None


class MLP(Module):
    """
    Feedforward network (MLP block).
    Typically expands dimension, applies non-linearity, then projects back.
    """
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        # Expansion and contraction layers
        self.gate_proj = Linear(config.d_model, config.d_ff, bias=config.use_bias)
        self.up_proj = Linear(config.d_model, config.d_ff, bias=config.use_bias)
        self.down_proj = Linear(config.d_ff, config.d_model, bias=config.use_bias)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        SwiGLU activation: Swish(xW) * (xV)
        Where Swish(x) = x * sigmoid(x)
        """
        if self.config.activation == 'swiglu':
            # SwiGLU: more efficient and performs well
            gate = self.gate_proj(x).silu() if hasattr(self.gate_proj(x), 'silu') else self._swish(self.gate_proj(x))
            up = self.up_proj(x)
            hidden = gate * up
        elif self.config.activation == 'gelu':
            hidden = self.gate_proj(x).gelu()
        else:  # relu
            hidden = self.gate_proj(x).relu()
        
        out = self.down_proj(hidden)
        
        if self.training and self.config.dropout > 0:
            out = out.dropout(self.config.dropout, training=True)
        
        return out
    
    def _swish(self, x: Tensor) -> Tensor:
        """Swish activation: x * sigmoid(x)"""
        return x * (1 / (1 + np.exp(-x.data)))


class TransformerBlock(Module):
    """
    Complete Transformer block with pre-normalization.
    Architecture: Norm -> Attention -> Residual -> Norm -> MLP -> Residual
    """
    
    def __init__(self, config: GPTConfig, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        
        # Normalization layers (pre-norm architecture)
        if config.norm_type == 'rmsnorm':
            self.norm1 = RMSNorm(config.d_model)
            self.norm2 = RMSNorm(config.d_model)
        else:
            self.norm1 = LayerNorm(config.d_model)
            self.norm2 = LayerNorm(config.d_model)
        
        # Attention and MLP
        self.attn = MultiHeadAttention(config)
        self.mlp = MLP(config)
    
    def forward(self, x: Tensor, attention_mask: Optional[np.ndarray] = None,
                store_attention: bool = False) -> Tuple[Tensor, Optional[np.ndarray]]:
        """
        Forward pass with residual connections.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            attention_mask: Optional attention mask
            store_attention: Whether to store attention weights
        
        Returns:
            output: (batch_size, seq_len, d_model)
            attention_weights: Optional attention matrix
        """
        # Self-attention with residual
        normed = self.norm1(x)
        attn_out, attn_weights = self.attn(normed, attention_mask, store_attention)
        x = x + attn_out  # Residual connection
        
        # MLP with residual
        normed = self.norm2(x)
        mlp_out = self.mlp(normed)
        x = x + mlp_out  # Residual connection
        
        return x, attn_weights


class MicroGPT(Module):
    """
    Complete GPT model for educational purposes.
    
    This is a from-scratch implementation that students can fully understand
    and experiment with.
    """
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_emb = Embedding(config.vocab_size, config.d_model)
        
        # Positional embeddings (learned)
        self.pos_emb = Embedding(config.max_seq_len, config.d_model)
        
        # Transformer blocks
        self.blocks = [TransformerBlock(config, i) for i in range(config.num_layers)]
        
        # Final normalization
        if config.norm_type == 'rmsnorm':
            self.norm_f = RMSNorm(config.d_model)
        else:
            self.norm_f = LayerNorm(config.d_model)
        
        # Language modeling head
        self.lm_head = Linear(config.d_model, config.vocab_size, bias=False)
        
        # Tie weights (sharing between token embedding and output)
        if config.tie_weights:
            self.lm_head.weight = self.token_emb.weight
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values for stability."""
        for module in self.modules():
            if isinstance(module, Linear):
                # Small initialization for output layer
                if module is self.lm_head:
                    module.weight.data *= 0.01
    
    def forward(self, input_ids: np.ndarray, targets: Optional[np.ndarray] = None,
                return_attention: bool = False) -> Tuple[Tensor, Optional[Tensor], Optional[List[np.ndarray]]]:
        """
        Forward pass through the entire model.
        
        Args:
            input_ids: Token IDs (batch_size, seq_len)
            targets: Target token IDs for loss computation (batch_size, seq_len)
            return_attention: Whether to return attention weights
        
        Returns:
            logits: Output logits (batch_size, seq_len, vocab_size)
            loss: Optional cross-entropy loss
            attention_weights: Optional list of attention matrices
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        token_embeds = self.token_emb(input_ids)
        
        # Positional embeddings
        positions = np.arange(seq_len)
        pos_embeds = self.pos_emb(positions)
        
        # Combine embeddings
        x = token_embeds + pos_embeds
        
        # Pass through transformer blocks
        attention_weights = [] if return_attention else None
        for block in self.blocks:
            x, attn = block(x, store_attention=return_attention)
            if return_attention and attn is not None:
                attention_weights.append(attn)
        
        # Final normalization
        x = self.norm_f(x)
        
        # Project to vocabulary
        logits = self.lm_head(x)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            # Reshape for cross-entropy: (batch*seq, vocab) and (batch*seq,)
            logits_flat = logits.reshape(-1, self.config.vocab_size)
            targets_flat = targets.reshape(-1)
            loss = cross_entropy_loss(logits_flat, targets_flat)
        
        return logits, loss, attention_weights
    
    def generate(self, input_ids: np.ndarray, max_new_tokens: int = 100,
                 temperature: float = 1.0, top_k: Optional[int] = None,
                 top_p: Optional[float] = None, 
                 repetition_penalty: float = 1.0) -> np.ndarray:
        """
        Generate text autoregressively.
        
        Args:
            input_ids: Starting token IDs (batch_size, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Limit sampling to top k tokens
            top_p: Nucleus sampling threshold
            repetition_penalty: Penalty for repeating tokens
        
        Returns:
            Generated token IDs (batch_size, seq_len + max_new_tokens)
        """
        self.eval()
        
        for _ in range(max_new_tokens):
            # Crop to max context length
            input_crop = input_ids[:, -self.config.max_seq_len:]
            
            # Forward pass
            logits, _, _ = self.forward(input_crop)
            
            # Get logits for last token
            next_token_logits = logits.data[:, -1, :] / temperature
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for batch_idx in range(input_ids.shape[0]):
                    for token_id in set(input_ids[batch_idx]):
                        next_token_logits[batch_idx, token_id] /= repetition_penalty
            
            # Top-k filtering
            if top_k is not None:
                indices_to_remove = next_token_logits < np.topk(next_token_logits, top_k, axis=-1)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits = np.sort(next_token_logits, axis=-1)[:, ::-1]
                sorted_indices = np.argsort(next_token_logits, axis=-1)[:, ::-1]
                cumulative_probs = np.cumsum(np.exp(sorted_logits) / np.sum(np.exp(sorted_logits), axis=-1, keepdims=True), axis=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 0] = False  # Keep at least one token
                
                for batch_idx in range(next_token_logits.shape[0]):
                    indices_to_remove = sorted_indices[batch_idx][sorted_indices_to_remove[batch_idx]]
                    next_token_logits[batch_idx, indices_to_remove] = float('-inf')
            
            # Sample from distribution
            probs = np.exp(next_token_logits) / np.sum(np.exp(next_token_logits), axis=-1, keepdims=True)
            next_token = np.array([np.random.choice(self.config.vocab_size, p=p) for p in probs])
            
            # Append to sequence
            input_ids = np.concatenate([input_ids, next_token[:, None]], axis=1)
        
        return input_ids
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration as dictionary."""
        return {
            'vocab_size': self.config.vocab_size,
            'max_seq_len': self.config.max_seq_len,
            'd_model': self.config.d_model,
            'num_layers': self.config.num_layers,
            'num_heads': self.config.num_heads,
            'd_ff': self.config.d_ff,
            'dropout': self.config.dropout,
            'activation': self.config.activation,
            'norm_type': self.config.norm_type,
            'tie_weights': self.config.tie_weights,
        }
