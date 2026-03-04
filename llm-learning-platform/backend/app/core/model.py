"""
GPT Model Architecture

Complete GPT-style decoder-only transformer built from scratch.
Supports multiple model sizes from Nano to XL, with full training
and inference capabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.core.tensor import Tensor
from app.core.embeddings import EmbeddingLayer, PositionalEncodingType, InitStrategy
from app.core.attention import MultiHeadAttention, AttentionType
from app.core.transformer import (
    TransformerBlock,
    NormType,
    NormPlacement,
    ActivationType,
    RMSNorm,
    LayerNorm,
)


# ─── Configuration ───────────────────────────────────────────

@dataclass
class GPTConfig:
    vocab_size: int = 256
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 4
    d_ff: int = 512
    max_seq_len: int = 256
    dropout: float = 0.1
    norm_type: str = "rmsnorm"
    norm_placement: str = "pre"
    activation: str = "gelu"
    attention_type: str = "full"
    positional_encoding: str = "sinusoidal"
    use_bias: bool = True
    tie_weights: bool = True

    @property
    def num_parameters(self) -> int:
        """Estimate total parameter count."""
        emb = self.vocab_size * self.d_model
        pos = self.max_seq_len * self.d_model
        # Per block: attention (4 * d_model^2) + MLP (varies) + norms
        if self.activation == "swiglu":
            mlp_per_block = 3 * self.d_model * self.d_ff
        else:
            mlp_per_block = 2 * self.d_model * self.d_ff
        attn_per_block = 4 * self.d_model ** 2
        norm_per_block = 2 * self.d_model
        block_total = (attn_per_block + mlp_per_block + norm_per_block) * self.num_layers
        head = self.d_model * self.vocab_size if not self.tie_weights else 0
        return emb + pos + block_total + head


# ─── Preset Configurations ──────────────────────────────────

PRESET_CONFIGS = {
    "nano": GPTConfig(
        vocab_size=256, d_model=64, num_heads=2, num_layers=2,
        d_ff=256, max_seq_len=128, dropout=0.1,
    ),
    "micro": GPTConfig(
        vocab_size=512, d_model=128, num_heads=4, num_layers=4,
        d_ff=512, max_seq_len=256, dropout=0.1,
    ),
    "mini": GPTConfig(
        vocab_size=1024, d_model=256, num_heads=8, num_layers=6,
        d_ff=1024, max_seq_len=512, dropout=0.1,
    ),
    "small": GPTConfig(
        vocab_size=8192, d_model=512, num_heads=8, num_layers=8,
        d_ff=2048, max_seq_len=1024, dropout=0.1,
    ),
    "medium": GPTConfig(
        vocab_size=16384, d_model=768, num_heads=12, num_layers=12,
        d_ff=3072, max_seq_len=1024, dropout=0.1,
    ),
    "large": GPTConfig(
        vocab_size=32000, d_model=1024, num_heads=16, num_layers=24,
        d_ff=4096, max_seq_len=2048, dropout=0.1,
    ),
}


# ─── GPT Model ───────────────────────────────────────────────

class MicroGPT:
    """
    Complete GPT-style decoder-only transformer.

    Built entirely from custom components for educational transparency.
    Every layer is inspectable and visualizable.
    """

    def __init__(self, config: GPTConfig):
        self.config = config

        # Token + positional embeddings
        self.embedding = EmbeddingLayer(
            vocab_size=config.vocab_size,
            embedding_dim=config.d_model,
            max_seq_len=config.max_seq_len,
            init_strategy=InitStrategy.NORMAL,
            positional_encoding=PositionalEncodingType(config.positional_encoding),
            dropout=config.dropout,
        )

        # Transformer blocks
        self.blocks: List[TransformerBlock] = []
        for _ in range(config.num_layers):
            block = TransformerBlock(
                d_model=config.d_model,
                num_heads=config.num_heads,
                d_ff=config.d_ff,
                norm_type=NormType(config.norm_type),
                norm_placement=NormPlacement(config.norm_placement),
                activation=ActivationType(config.activation),
                attention_type=AttentionType(config.attention_type),
                dropout=config.dropout,
                use_bias=config.use_bias,
                num_layers=config.num_layers,
            )
            self.blocks.append(block)

        # Final layer norm
        if config.norm_type == "rmsnorm":
            self.final_norm = RMSNorm(config.d_model)
        else:
            self.final_norm = LayerNorm(config.d_model)

        # Language model head (output projection)
        if config.tie_weights:
            self.lm_head = self.embedding.weight  # weight tying
        else:
            scale = np.sqrt(2.0 / config.d_model).astype(np.float32)
            self.lm_head = Tensor(
                np.random.randn(config.d_model, config.vocab_size).astype(np.float32) * scale,
                requires_grad=True,
            )

        self._training = True

    # ─── Forward Pass ────────────────────────────────────────

    def forward(
        self,
        token_ids: np.ndarray,
        targets: Optional[np.ndarray] = None,
        store_intermediates: bool = False,
    ) -> Dict:
        """
        Forward pass through the full model.

        Args:
            token_ids: (batch, seq_len) integer token IDs
            targets: optional (batch, seq_len) for loss computation
            store_intermediates: whether to save layer outputs
        Returns:
            dict with 'logits', optionally 'loss' and 'intermediates'
        """
        intermediates: Dict[str, object] = {}

        if token_ids.ndim == 1:
            token_ids = token_ids[np.newaxis, :]

        batch_size, seq_len = token_ids.shape

        # Embeddings
        x = self.embedding.forward(token_ids)
        if isinstance(x, Tensor):
            x = x.data

        if store_intermediates:
            intermediates["embeddings"] = x.copy()

        # Causal mask
        mask = MultiHeadAttention.create_causal_mask(seq_len)

        # Transformer blocks
        for i, block in enumerate(self.blocks):
            result = block.forward(x, mask, store_intermediates=store_intermediates)
            x = result.output
            if store_intermediates and result.intermediates:
                intermediates[f"block_{i}"] = result.intermediates

        # Final norm
        x = self.final_norm.forward(x)
        if store_intermediates:
            intermediates["final_norm"] = x.copy()

        # Logits
        if self.config.tie_weights:
            logits = x @ self.lm_head.data.T
        else:
            logits = x @ self.lm_head.data

        result_dict: Dict = {"logits": logits}

        # Loss
        if targets is not None:
            logits_tensor = Tensor(logits, requires_grad=True)
            loss_tensor = logits_tensor.cross_entropy(targets)
            result_dict["loss"] = float(loss_tensor.data)
            result_dict["loss_tensor"] = loss_tensor

        if store_intermediates:
            result_dict["intermediates"] = intermediates

        return result_dict

    # ─── Text Generation ────────────────────────────────────

    def generate(
        self,
        prompt_ids: np.ndarray,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Autoregressive text generation with sampling strategies.

        Returns:
            (generated_ids, step_metadata) — the full sequence and
            per-step information for visualization.
        """
        self.set_training(False)

        if prompt_ids.ndim == 1:
            prompt_ids = prompt_ids[np.newaxis, :]

        current = prompt_ids.copy()
        step_metadata: List[Dict] = []

        for step in range(max_new_tokens):
            # Truncate to max sequence length
            context = current[:, -self.config.max_seq_len :]

            result = self.forward(context)
            logits = result["logits"][:, -1, :]  # last token logits

            # Temperature scaling
            if temperature != 1.0:
                logits = logits / temperature

            # Top-k filtering
            if top_k > 0:
                top_k_val = min(top_k, logits.shape[-1])
                threshold = np.sort(logits, axis=-1)[:, -top_k_val : -top_k_val + 1]
                logits[logits < threshold] = -1e9

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits = np.sort(logits, axis=-1)[:, ::-1]
                sorted_indices = np.argsort(logits, axis=-1)[:, ::-1]
                probs = np.exp(sorted_logits - sorted_logits.max(axis=-1, keepdims=True))
                probs = probs / probs.sum(axis=-1, keepdims=True)
                cumulative = np.cumsum(probs, axis=-1)
                mask = cumulative - probs > top_p
                for b in range(logits.shape[0]):
                    remove_indices = sorted_indices[b][mask[b]]
                    logits[b, remove_indices] = -1e9

            # Sample
            probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
            probs = probs / probs.sum(axis=-1, keepdims=True)

            next_token = np.array(
                [np.random.choice(len(p), p=p) for p in probs]
            ).reshape(-1, 1)

            current = np.concatenate([current, next_token], axis=1)

            step_metadata.append({
                "step": step,
                "token_id": int(next_token[0, 0]),
                "probabilities": probs[0].tolist(),
                "top_5": [
                    {"id": int(idx), "prob": float(probs[0, idx])}
                    for idx in np.argsort(probs[0])[-5:][::-1]
                ],
            })

        self.set_training(True)
        return current, step_metadata

    # ─── Training Utilities ──────────────────────────────────

    def parameters(self) -> List[Tensor]:
        params: List[Tensor] = []
        params.extend(self.embedding.parameters())
        for block in self.blocks:
            params.extend(block.parameters())
        params.extend(self.final_norm.parameters())
        if not self.config.tie_weights:
            params.append(self.lm_head)
        return params

    def num_parameters(self) -> int:
        return sum(p.data.size for p in self.parameters())

    def set_training(self, training: bool):
        self._training = training
        for block in self.blocks:
            block.set_training(training)

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def get_config_summary(self) -> Dict:
        return {
            "vocab_size": self.config.vocab_size,
            "d_model": self.config.d_model,
            "num_heads": self.config.num_heads,
            "num_layers": self.config.num_layers,
            "d_ff": self.config.d_ff,
            "max_seq_len": self.config.max_seq_len,
            "num_parameters": self.num_parameters(),
            "norm_type": self.config.norm_type,
            "activation": self.config.activation,
            "attention_type": self.config.attention_type,
            "positional_encoding": self.config.positional_encoding,
        }

    # ─── Checkpoint Save/Load ────────────────────────────────

    def save_checkpoint(self, path: str):
        """Save model weights to file."""
        state = {
            "config": self.config.__dict__,
            "weights": {
                f"param_{i}": p.data for i, p in enumerate(self.parameters())
            },
        }
        np.savez_compressed(path, **{k: v for k, v in state["weights"].items()},
                           config=np.array([str(state["config"])]))

    def load_checkpoint(self, path: str):
        """Load model weights from file."""
        data = np.load(path, allow_pickle=True)
        params = self.parameters()
        for i, p in enumerate(params):
            key = f"param_{i}"
            if key in data:
                p.data = data[key].astype(np.float32)
