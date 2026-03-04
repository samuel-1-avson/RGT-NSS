"""
MicroGPT Model — PyTorch GPU-accelerated Causal Language Model

A real GPT-family language model built with PyTorch nn.Module.
Supports GPU training, inference, generation with sampling strategies,
and checkpoint save/load.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.core.device import get_device
from app.core.embeddings import EmbeddingLayer, PositionalEncodingType
from app.core.transformer import TransformerBlock, NormType, NormPlacement, ActivationType, RMSNorm
from app.core.attention import AttentionType


@dataclass
class GPTConfig:
    """Configuration for MicroGPT model."""
    vocab_size: int = 256
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 4
    d_ff: int = 512
    max_seq_len: int = 256
    dropout: float = 0.1
    norm_type: str = "rmsnorm"
    activation: str = "gelu"
    use_bias: bool = True

    @property
    def num_parameters(self) -> int:
        """Estimate total parameter count."""
        emb = self.vocab_size * self.d_model
        pos_emb = self.max_seq_len * self.d_model
        attn = 4 * self.d_model * self.d_model  # Q, K, V, O projections
        mlp = 2 * self.d_model * self.d_ff
        per_layer = attn + mlp + 2 * self.d_model  # + norms
        lm_head = self.vocab_size * self.d_model
        return emb + pos_emb + self.num_layers * per_layer + lm_head


# ─── Preset Configurations ──────────────────────────────────

PRESET_CONFIGS = {
    "nano": GPTConfig(
        vocab_size=256, d_model=64, num_heads=2, num_layers=2,
        d_ff=256, max_seq_len=128, dropout=0.1,
    ),
    "micro": GPTConfig(
        vocab_size=256, d_model=128, num_heads=4, num_layers=4,
        d_ff=512, max_seq_len=256, dropout=0.1,
    ),
    "small": GPTConfig(
        vocab_size=8192, d_model=256, num_heads=8, num_layers=6,
        d_ff=1024, max_seq_len=512, dropout=0.1,
    ),
    "medium": GPTConfig(
        vocab_size=16384, d_model=512, num_heads=8, num_layers=8,
        d_ff=2048, max_seq_len=1024, dropout=0.1,
    ),
}


class MicroGPT(nn.Module):
    """
    GPU-accelerated GPT language model.

    A real transformer decoder with:
    - Token + positional embeddings
    - N stacked transformer blocks (attention + MLP + norms)
    - Tied LM head (weight sharing with embeddings)
    - GPU training and inference
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.device = get_device()

        # Embedding layer
        self.embedding = EmbeddingLayer(
            vocab_size=config.vocab_size,
            embedding_dim=config.d_model,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
        )

        # Transformer blocks
        norm_type = NormType.RMSNORM if config.norm_type == "rmsnorm" else NormType.LAYERNORM
        activation = {
            "gelu": ActivationType.GELU,
            "relu": ActivationType.RELU,
            "swiglu": ActivationType.SWIGLU,
            "silu": ActivationType.SILU,
        }.get(config.activation, ActivationType.GELU)

        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                num_heads=config.num_heads,
                d_ff=config.d_ff,
                norm_type=norm_type,
                norm_placement=NormPlacement.PRE,
                activation=activation,
                dropout=config.dropout,
                use_bias=config.use_bias,
            )
            for _ in range(config.num_layers)
        ])

        # Final norm + LM head
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying (embedding ↔ lm_head)
        self.lm_head.weight = self.embedding.token_embedding.weight

        # Move to device
        self.to(self.device)
        self._apply_init()

    def _apply_init(self):
        """Apply scaled initialization."""
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.normal_(p, std=0.02)
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(
        self,
        token_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        store_intermediates: bool = False,
    ) -> Dict:
        """
        Forward pass through the model.

        Args:
            token_ids: (batch, seq_len) integer token IDs
            targets: (batch, seq_len) target token IDs for loss
            store_intermediates: save layer outputs for visualization
        Returns:
            Dict with 'logits', optionally 'loss' and 'intermediates'
        """
        if isinstance(token_ids, np.ndarray):
            token_ids = torch.from_numpy(token_ids).long()
        token_ids = token_ids.to(self.device)

        B, S = token_ids.shape
        mask = torch.tril(torch.ones(S, S, device=self.device))

        # Embeddings
        x = self.embedding(token_ids)  # (B, S, D)

        # Transformer blocks
        intermediates = {}
        for i, layer in enumerate(self.layers):
            result = layer(x, mask=mask, store_intermediates=store_intermediates)
            x = result.output
            if store_intermediates and result.intermediates:
                intermediates[f"layer_{i}"] = result.intermediates

        # Final norm + LM head
        x = self.final_norm(x)
        logits = self.lm_head(x)  # (B, S, vocab_size)

        result = {"logits": logits.detach().cpu().numpy()}

        if store_intermediates:
            result["intermediates"] = intermediates

        # Compute loss if targets provided
        if targets is not None:
            if isinstance(targets, np.ndarray):
                targets = torch.from_numpy(targets).long()
            targets = targets.to(self.device)

            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
            )
            result["loss"] = loss.item()
            result["_loss_tensor"] = loss  # Keep for backprop

        return result

    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9,
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Autoregressive text generation with sampling.

        Returns:
            (generated_ids, step_info) — generated token array and per-step metadata
        """
        self.eval()

        if isinstance(prompt, np.ndarray):
            prompt = torch.from_numpy(prompt).long()
        prompt = prompt.to(self.device)

        generated = prompt.clone()
        step_info = []

        with torch.no_grad():
            for step in range(max_new_tokens):
                # Truncate to max_seq_len
                context = generated[:, -self.config.max_seq_len:]
                result = self._forward_for_generation(context)
                logits = result[:, -1, :]  # Last position

                # Temperature scaling
                logits = logits / max(temperature, 1e-8)

                # Top-k filtering
                if top_k > 0:
                    k = min(top_k, logits.size(-1))
                    top_k_vals, _ = torch.topk(logits, k)
                    logits[logits < top_k_vals[:, -1:]] = float("-inf")

                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    cutoff = cumulative_probs > top_p
                    cutoff[:, 1:] = cutoff[:, :-1].clone()
                    cutoff[:, 0] = False
                    indices_to_remove = cutoff.scatter(1, sorted_indices, cutoff)
                    logits[indices_to_remove] = float("-inf")

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)

                step_info.append({
                    "step": step,
                    "token_id": next_token.item(),
                    "top_probs": probs[0].topk(5).values.cpu().numpy().tolist(),
                    "top_tokens": probs[0].topk(5).indices.cpu().numpy().tolist(),
                })

        self.train()
        return generated.cpu().numpy(), step_info

    def _forward_for_generation(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Lightweight forward pass for generation (returns logits tensor directly)."""
        B, S = token_ids.shape
        mask = torch.tril(torch.ones(S, S, device=self.device))
        x = self.embedding(token_ids)
        for layer in self.layers:
            result = layer(x, mask=mask, store_intermediates=False)
            x = result.output
        x = self.final_norm(x)
        return self.lm_head(x)

    def set_training(self, training: bool):
        """Toggle training/eval mode."""
        if training:
            self.train()
        else:
            self.eval()

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            "model_state_dict": self.state_dict(),
            "config": self.config,
        }, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.load_state_dict(checkpoint["model_state_dict"])

    def parameters_count(self) -> Dict:
        """Get parameter count breakdown."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
            "size_mb": round(total * 4 / 1e6, 2),  # float32
        }

    def forward_with_intermediates(self, token_ids) -> Dict:
        """Forward pass that returns all intermediate layer outputs."""
        return self.forward(token_ids, store_intermediates=True)
