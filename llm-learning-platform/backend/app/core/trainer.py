"""
Training Engine — PyTorch GPU-accelerated Training Loop

Real training loop with AdamW optimizer, gradient clipping,
learning rate scheduling, and WebSocket progress streaming.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, StepLR

from app.core.device import get_device, get_device_info
from app.core.model import MicroGPT, GPTConfig, PRESET_CONFIGS


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    batch_size: int = 4
    max_steps: int = 100
    warmup_steps: int = 10
    grad_clip: float = 1.0
    optimizer: str = "adamw"  # adam, adamw, sgd
    scheduler: str = "cosine"  # cosine, step, onecycle, none
    eval_interval: int = 10
    log_interval: int = 5
    accumulation_steps: int = 1


@dataclass
class TrainStepResult:
    """Result of a single training step."""
    step: int
    loss: float
    learning_rate: float
    grad_norm: float = 0.0
    tokens_per_sec: float = 0.0
    gpu_memory_mb: float = 0.0


class TextDataset:
    """Simple character-level dataset for training demonstrations."""

    def __init__(self, text: str, seq_len: int, vocab_size: int = 256):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        # Character-level tokenization (simple but real)
        self.token_ids = torch.tensor(
            [ord(c) % vocab_size for c in text], dtype=torch.long, device=get_device()
        )

    def __len__(self) -> int:
        return max(1, len(self.token_ids) - self.seq_len - 1)

    def get_batch(self, batch_size: int) -> tuple:
        """Get a random batch of (input, target) pairs."""
        max_start = len(self.token_ids) - self.seq_len - 1
        if max_start <= 0:
            # If text is too short, pad with repeats
            padded = self.token_ids.repeat(self.seq_len * 2 // len(self.token_ids) + 1)
            self.token_ids = padded
            max_start = len(self.token_ids) - self.seq_len - 1

        indices = torch.randint(0, max_start, (batch_size,))
        x = torch.stack([self.token_ids[i : i + self.seq_len] for i in indices])
        y = torch.stack([self.token_ids[i + 1 : i + self.seq_len + 1] for i in indices])
        return x, y


# ─── Training Corpus ─────────────────────────────────────────

TRAINING_CORPUS = """
The transformer architecture has revolutionized natural language processing.
Self-attention mechanisms allow the model to weigh the importance of different
parts of the input sequence when producing an output. Unlike recurrent neural
networks, transformers process all positions in parallel, making them much
faster to train on modern hardware.

Key components of a transformer include multi-head attention, feed-forward
neural networks, layer normalization, and residual connections. The attention
mechanism computes query, key, and value projections, then uses scaled
dot-product attention to determine how much each position should attend to
every other position.

Language models like GPT use decoder-only transformer architectures. They
are trained with a causal language modeling objective, predicting the next
token given all previous tokens. This autoregressive approach enables text
generation by sampling one token at a time.

Training involves minimizing the cross-entropy loss between the model's
predicted token probabilities and the actual next tokens in the training
data. Techniques like gradient clipping, learning rate warmup, and weight
decay help stabilize the training process.

Modern large language models are trained on massive datasets using distributed
training across many GPUs. Techniques like data parallelism, model parallelism,
and pipeline parallelism allow training models with billions of parameters.
Fine-tuning with RLHF (Reinforcement Learning from Human Feedback) aligns
the model's outputs with human preferences and values.
"""


class Trainer:
    """
    GPU-accelerated model trainer with real PyTorch optimization.

    Supports AdamW, gradient clipping, LR scheduling, and
    step-by-step training visualization.
    """

    def __init__(
        self,
        model: Optional[MicroGPT] = None,
        config: Optional[TrainingConfig] = None,
        training_text: Optional[str] = None,
    ):
        self.config = config or TrainingConfig()

        # Create model if not provided
        if model is None:
            model_config = PRESET_CONFIGS["nano"]
            model = MicroGPT(model_config)
        self.model = model
        self.model.train()

        # Setup optimizer
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # Setup dataset
        text = training_text or TRAINING_CORPUS
        self.dataset = TextDataset(
            text,
            seq_len=min(self.model.config.max_seq_len, 64),
            vocab_size=self.model.config.vocab_size,
        )

        self.step_count = 0
        self.history: List[TrainStepResult] = []

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with weight decay separation."""
        # Separate weight decay for non-bias/norm parameters
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "norm" in name or "embedding" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        if self.config.optimizer == "adamw":
            return AdamW(param_groups, lr=self.config.learning_rate, betas=(0.9, 0.95))
        elif self.config.optimizer == "adam":
            return Adam(param_groups, lr=self.config.learning_rate)
        elif self.config.optimizer == "sgd":
            return SGD(param_groups, lr=self.config.learning_rate, momentum=0.9)
        else:
            return AdamW(param_groups, lr=self.config.learning_rate)

    def _create_scheduler(self):
        """Create learning rate scheduler."""
        if self.config.scheduler == "cosine":
            return CosineAnnealingLR(self.optimizer, T_max=self.config.max_steps)
        elif self.config.scheduler == "step":
            return StepLR(self.optimizer, step_size=30, gamma=0.5)
        elif self.config.scheduler == "onecycle":
            return OneCycleLR(
                self.optimizer,
                max_lr=self.config.learning_rate * 10,
                total_steps=self.config.max_steps,
            )
        return None

    def train_step(self) -> TrainStepResult:
        """Execute a single training step with real gradient computation."""
        self.model.train()

        # Get batch
        x, y = self.dataset.get_batch(self.config.batch_size)

        # Forward pass
        result = self.model.forward(x, targets=y)
        loss = result["_loss_tensor"]

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.grad_clip
        ).item()

        # Optimizer step
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()

        self.step_count += 1

        # GPU memory tracking
        gpu_mem = 0.0
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / 1e6

        lr = self.optimizer.param_groups[0]["lr"]

        step_result = TrainStepResult(
            step=self.step_count,
            loss=loss.item(),
            learning_rate=lr,
            grad_norm=round(grad_norm, 4),
            gpu_memory_mb=round(gpu_mem, 1),
        )

        self.history.append(step_result)
        return step_result

    def train(
        self,
        num_steps: Optional[int] = None,
        callback: Optional[Callable] = None,
    ) -> List[Dict]:
        """
        Run training for N steps.

        Args:
            num_steps: override config.max_steps
            callback: called after each step with TrainStepResult
        Returns:
            List of step results as dicts
        """
        steps = num_steps or self.config.max_steps
        results = []

        for _ in range(steps):
            step_result = self.train_step()
            result_dict = {
                "step": step_result.step,
                "loss": round(step_result.loss, 6),
                "learning_rate": step_result.learning_rate,
                "grad_norm": step_result.grad_norm,
                "gpu_memory_mb": step_result.gpu_memory_mb,
            }
            results.append(result_dict)
            if callback:
                callback(step_result)

        return results

    def get_training_summary(self) -> Dict:
        """Get summary of training progress."""
        if not self.history:
            return {"status": "not_started"}

        losses = [h.loss for h in self.history]
        return {
            "total_steps": len(self.history),
            "final_loss": round(losses[-1], 6),
            "best_loss": round(min(losses), 6),
            "initial_loss": round(losses[0], 6),
            "loss_reduction": round((losses[0] - losses[-1]) / max(losses[0], 1e-8) * 100, 1),
            "device": str(get_device()),
            "model_params": self.model.parameters_count(),
            "config": {
                "lr": self.config.learning_rate,
                "optimizer": self.config.optimizer,
                "scheduler": self.config.scheduler,
                "batch_size": self.config.batch_size,
                "grad_clip": self.config.grad_clip,
            },
        }


def quick_train(
    preset: str = "nano",
    num_steps: int = 50,
    learning_rate: float = 3e-4,
    training_text: Optional[str] = None,
) -> Dict:
    """Quick-start training with sensible defaults."""
    model_config = PRESET_CONFIGS.get(preset, PRESET_CONFIGS["nano"])
    model = MicroGPT(model_config)

    train_config = TrainingConfig(
        learning_rate=learning_rate,
        max_steps=num_steps,
        batch_size=4,
    )

    trainer = Trainer(model=model, config=train_config, training_text=training_text)
    results = trainer.train(num_steps=num_steps)

    return {
        "model_preset": preset,
        "training_results": results,
        "summary": trainer.get_training_summary(),
        "device_info": get_device_info(),
    }
