"""
PyTorch Training Engine with GPU acceleration.
Mirrors the custom TrainingEngine API but uses PyTorch for real performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
from typing import Optional, Dict, Any, List, Generator, Callable
from dataclasses import dataclass, field
from pathlib import Path

from app.models.pytorch_gpt import PyTorchGPT, PyTorchGPTConfig, get_device


# =============================================================================
# TRAINING CONFIG
# =============================================================================

@dataclass
class PyTorchTrainingConfig:
    """Training configuration for PyTorch models."""
    batch_size: int = 32
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 100
    max_steps: int = 1000
    grad_clip: float = 1.0
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.999)
    seq_length: int = 256
    eval_interval: int = 100
    eval_steps: int = 10
    checkpoint_interval: int = 1000
    checkpoint_dir: str = "./checkpoints"
    log_interval: int = 10
    use_amp: bool = True          # Automatic Mixed Precision
    gradient_accumulation_steps: int = 1


# =============================================================================
# TRAINING METRICS
# =============================================================================

@dataclass
class TrainingMetrics:
    """Metrics from a training step — compatible with existing WebSocket format."""
    step: int
    epoch: int
    loss: float
    perplexity: float
    learning_rate: float
    grad_norm: float
    tokens_per_sec: float
    time_elapsed: float
    time_remaining: Optional[float] = None
    gpu_memory_mb: Optional[float] = None
    gpu_utilization: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "epoch": self.epoch,
            "loss": round(self.loss, 6),
            "perplexity": round(self.perplexity, 4),
            "learning_rate": self.learning_rate,
            "grad_norm": round(self.grad_norm, 6),
            "tokens_per_sec": round(self.tokens_per_sec, 1),
            "time_elapsed": round(self.time_elapsed, 2),
            "time_remaining": round(self.time_remaining, 2) if self.time_remaining else None,
            "gpu_memory_mb": round(self.gpu_memory_mb, 1) if self.gpu_memory_mb else None,
            "gpu_utilization": self.gpu_utilization,
        }


# =============================================================================
# DATASET (PyTorch-compatible)
# =============================================================================

class TextDatasetPyTorch(torch.utils.data.Dataset):
    """PyTorch Dataset wrapper for text data."""

    def __init__(self, data: np.ndarray, seq_length: int):
        self.data = torch.from_numpy(data).long()
        self.seq_length = seq_length

    def __len__(self):
        return max(1, len(self.data) - self.seq_length - 1)

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_length]
        y = self.data[idx + 1: idx + self.seq_length + 1]
        return x, y


# =============================================================================
# LEARNING RATE SCHEDULER
# =============================================================================

def get_lr(step: int, config: PyTorchTrainingConfig) -> float:
    """Cosine decay with linear warmup."""
    if step < config.warmup_steps:
        return config.learning_rate * step / max(1, config.warmup_steps)
    if step >= config.max_steps:
        return config.min_learning_rate
    # Cosine decay
    decay_ratio = (step - config.warmup_steps) / max(1, config.max_steps - config.warmup_steps)
    coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))
    return config.min_learning_rate + coeff * (config.learning_rate - config.min_learning_rate)


# =============================================================================
# TRAINING ENGINE
# =============================================================================

class PyTorchTrainingEngine:
    """GPU-accelerated training engine for PyTorchGPT.

    Features:
    - Automatic Mixed Precision (AMP) for faster training
    - Gradient accumulation for effective larger batch sizes
    - Gradient clipping
    - Cosine LR schedule with warmup
    - Real-time metric reporting (WebSocket compatible)
    - Checkpointing with torch.save
    - GPU memory monitoring
    """

    def __init__(self, model: PyTorchGPT, config: PyTorchTrainingConfig):
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device

        # Optimizer — AdamW with weight decay
        # Separate weight decay for different parameter groups
        decay_params = []
        no_decay_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.dim() >= 2:  # Matrices get weight decay
                    decay_params.append(param)
                else:  # Biases, norms don't
                    no_decay_params.append(param)

        self.optimizer = torch.optim.AdamW([
            {"params": decay_params, "weight_decay": config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ], lr=config.learning_rate, betas=config.betas)

        # Mixed precision scaler
        self.scaler = torch.amp.GradScaler(enabled=config.use_amp and self.device.type == "cuda")

        # State
        self.current_step = 0
        self.current_epoch = 0
        self.best_loss = float("inf")
        self.is_training = False
        self._should_stop = False
        self.callbacks: List[Callable] = []
        self.metrics_history: List[TrainingMetrics] = []
        self.start_time = None

        # Ensure checkpoint dir exists
        os.makedirs(config.checkpoint_dir, exist_ok=True)

    def add_callback(self, callback: Callable):
        """Add a callback for metrics reporting."""
        self.callbacks.append(callback)

    def _get_gpu_stats(self) -> Dict[str, float]:
        """Get current GPU memory usage."""
        if self.device.type == "cuda":
            return {
                "gpu_memory_mb": torch.cuda.memory_allocated(self.device) / (1024 * 1024),
                "gpu_memory_reserved_mb": torch.cuda.memory_reserved(self.device) / (1024 * 1024),
                "gpu_memory_max_mb": torch.cuda.max_memory_allocated(self.device) / (1024 * 1024),
            }
        return {}

    def _get_gradient_norms(self) -> Dict[str, float]:
        """Get per-layer gradient norms for visualization."""
        grad_norms = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norms[name] = param.grad.norm().item()
        return grad_norms

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> TrainingMetrics:
        """Execute single training step with mixed precision."""
        self.model.train()
        step_start = time.time()

        # Update learning rate
        lr = get_lr(self.current_step, self.config)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        # Forward pass with mixed precision
        with torch.amp.autocast(device_type=self.device.type, enabled=self.config.use_amp):
            result = self.model(x)
            logits = result["logits"]
            # Cross-entropy loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
            )
            loss = loss / self.config.gradient_accumulation_steps

        # Backward pass
        self.scaler.scale(loss).backward()

        # Gradient accumulation
        if (self.current_step + 1) % self.config.gradient_accumulation_steps == 0:
            # Unscale for gradient clipping
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip
            ).item()

            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
        else:
            grad_norm = 0.0

        # Compute metrics
        actual_loss = loss.item() * self.config.gradient_accumulation_steps
        perplexity = min(float(np.exp(actual_loss)), 1e6)  # Clamp perplexity
        step_time = time.time() - step_start
        tokens = x.numel()
        tokens_per_sec = tokens / max(step_time, 1e-6)

        elapsed = time.time() - self.start_time if self.start_time else 0
        remaining = None
        if self.current_step > 0 and elapsed > 0:
            steps_remaining = self.config.max_steps - self.current_step
            time_per_step = elapsed / self.current_step
            remaining = steps_remaining * time_per_step

        gpu_stats = self._get_gpu_stats()

        metrics = TrainingMetrics(
            step=self.current_step,
            epoch=self.current_epoch,
            loss=actual_loss,
            perplexity=perplexity,
            learning_rate=lr,
            grad_norm=grad_norm,
            tokens_per_sec=tokens_per_sec,
            time_elapsed=elapsed,
            time_remaining=remaining,
            gpu_memory_mb=gpu_stats.get("gpu_memory_mb"),
        )

        self.metrics_history.append(metrics)
        if actual_loss < self.best_loss:
            self.best_loss = actual_loss

        self.current_step += 1

        # Notify callbacks
        for cb in self.callbacks:
            cb(metrics)

        return metrics

    @torch.no_grad()
    def evaluate(self, val_data: np.ndarray, max_steps: Optional[int] = None) -> float:
        """Evaluate model on validation data."""
        self.model.eval()
        dataset = TextDatasetPyTorch(val_data, self.config.seq_length)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=False, drop_last=True
        )
        total_loss = 0.0
        count = 0
        steps = max_steps or self.config.eval_steps

        for i, (x, y) in enumerate(loader):
            if i >= steps:
                break
            x, y = x.to(self.device), y.to(self.device)
            with torch.amp.autocast(device_type=self.device.type, enabled=self.config.use_amp):
                result = self.model(x)
                loss = F.cross_entropy(
                    result["logits"].view(-1, result["logits"].size(-1)),
                    y.view(-1),
                )
            total_loss += loss.item()
            count += 1

        self.model.train()
        return total_loss / max(count, 1)

    def train(
        self,
        train_data: np.ndarray,
        val_data: Optional[np.ndarray] = None,
    ) -> Generator[TrainingMetrics, None, None]:
        """
        Full training loop.

        Yields TrainingMetrics after each step for real-time WebSocket streaming.
        """
        self.is_training = True
        self._should_stop = False
        self.start_time = time.time()

        # Create dataset and dataloader
        dataset = TextDatasetPyTorch(train_data, self.config.seq_length)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=(self.device.type == "cuda"),
            num_workers=0,  # Keep 0 on Windows to avoid multiprocessing issues
        )

        epoch = 0
        while self.current_step < self.config.max_steps and not self._should_stop:
            self.current_epoch = epoch
            for x, y in loader:
                if self.current_step >= self.config.max_steps or self._should_stop:
                    break

                x, y = x.to(self.device), y.to(self.device)
                metrics = self.train_step(x, y)
                yield metrics

                # Checkpointing
                if self.current_step % self.config.checkpoint_interval == 0:
                    self.save_checkpoint()

            epoch += 1

        self.is_training = False

    def stop(self):
        """Stop training gracefully."""
        self._should_stop = True
        self.is_training = False

    def save_checkpoint(self, path: Optional[str] = None):
        """Save model checkpoint."""
        if path is None:
            path = os.path.join(
                self.config.checkpoint_dir,
                f"checkpoint_step_{self.current_step}.pt"
            )

        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "step": self.current_step,
            "epoch": self.current_epoch,
            "best_loss": self.best_loss,
            "config": self.config,
            "model_config": self.model.config,
        }, path)

        return path

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.current_step = checkpoint.get("step", 0)
        self.current_epoch = checkpoint.get("epoch", 0)
        self.best_loss = checkpoint.get("best_loss", float("inf"))

    def get_status(self) -> Dict[str, Any]:
        """Get current training status."""
        status = {
            "is_training": self.is_training,
            "current_step": self.current_step,
            "current_epoch": self.current_epoch,
            "best_loss": self.best_loss,
            "max_steps": self.config.max_steps,
            "progress": self.current_step / max(self.config.max_steps, 1),
            "device": str(self.device),
        }
        status.update(self._get_gpu_stats())
        return status
