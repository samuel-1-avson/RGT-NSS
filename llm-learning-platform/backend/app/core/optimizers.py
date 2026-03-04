"""
Optimizers and Learning Rate Schedulers

AdamW, SGD with momentum, and Lion optimizer, plus cosine,
warmup-cosine, and linear schedulers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from app.core.tensor import Tensor


# ─── Optimizers ──────────────────────────────────────────────

class AdamW:
    """
    AdamW optimizer (Loshchilov & Hutter, 2019).
    Decoupled weight decay regularization.
    """

    def __init__(
        self,
        params: List[Tensor],
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_count = 0

        # Moment estimates
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]

    def step(self):
        self.step_count += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            g = p.grad

            # Update biased moments
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g ** 2

            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.step_count)
            v_hat = self.v[i] / (1 - self.beta2 ** self.step_count)

            # Weight decay (decoupled)
            p.data -= self.lr * self.weight_decay * p.data

            # Parameter update
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()

    def get_state(self) -> Dict:
        return {
            "step": self.step_count,
            "lr": self.lr,
            "moments": [
                {"m_norm": float(np.linalg.norm(m)), "v_norm": float(np.linalg.norm(v))}
                for m, v in zip(self.m, self.v)
            ],
        }


class SGD:
    """SGD with optional momentum and Nesterov acceleration."""

    def __init__(
        self,
        params: List[Tensor],
        lr: float = 0.01,
        momentum: float = 0.0,
        nesterov: bool = False,
        weight_decay: float = 0.0,
    ):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = weight_decay
        self.step_count = 0
        self.velocity = [np.zeros_like(p.data) for p in params]

    def step(self):
        self.step_count += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            g = p.grad
            if self.weight_decay:
                g = g + self.weight_decay * p.data

            if self.momentum:
                self.velocity[i] = self.momentum * self.velocity[i] + g
                if self.nesterov:
                    update = g + self.momentum * self.velocity[i]
                else:
                    update = self.velocity[i]
            else:
                update = g

            p.data -= self.lr * update

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()


class Lion:
    """
    Lion optimizer (Chen et al., 2023).
    Uses sign of momentum for update — simpler and often faster than Adam.
    """

    def __init__(
        self,
        params: List[Tensor],
        lr: float = 1e-4,
        betas: tuple = (0.9, 0.99),
        weight_decay: float = 0.0,
    ):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.weight_decay = weight_decay
        self.step_count = 0
        self.m = [np.zeros_like(p.data) for p in params]

    def step(self):
        self.step_count += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            g = p.grad

            # Update direction: sign of interpolated momentum
            update = self.beta1 * self.m[i] + (1 - self.beta1) * g
            p.data -= self.lr * (np.sign(update) + self.weight_decay * p.data)

            # Update EMA
            self.m[i] = self.beta2 * self.m[i] + (1 - self.beta2) * g

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()


# ─── Learning Rate Schedulers ────────────────────────────────

class CosineScheduler:
    """Cosine annealing with optional warmup."""

    def __init__(
        self,
        optimizer,
        total_steps: int,
        warmup_steps: int = 0,
        min_lr: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.base_lr = optimizer.lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self.current_step = 0

    def step(self):
        self.current_step += 1
        lr = self.get_lr()
        self.optimizer.lr = lr

    def get_lr(self) -> float:
        if self.current_step < self.warmup_steps:
            # Linear warmup
            return self.base_lr * self.current_step / max(self.warmup_steps, 1)
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / max(
                self.total_steps - self.warmup_steps, 1
            )
            progress = min(progress, 1.0)
            return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1 + math.cos(math.pi * progress)
            )


class LinearScheduler:
    """Linear warmup then linear decay."""

    def __init__(
        self,
        optimizer,
        total_steps: int,
        warmup_steps: int = 0,
        min_lr: float = 0.0,
    ):
        self.optimizer = optimizer
        self.base_lr = optimizer.lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self.current_step = 0

    def step(self):
        self.current_step += 1
        lr = self.get_lr()
        self.optimizer.lr = lr

    def get_lr(self) -> float:
        if self.current_step < self.warmup_steps:
            return self.base_lr * self.current_step / max(self.warmup_steps, 1)
        else:
            progress = (self.current_step - self.warmup_steps) / max(
                self.total_steps - self.warmup_steps, 1
            )
            progress = min(progress, 1.0)
            return self.base_lr - (self.base_lr - self.min_lr) * progress


class ConstantScheduler:
    """Constant learning rate with optional warmup."""

    def __init__(self, optimizer, warmup_steps: int = 0):
        self.optimizer = optimizer
        self.base_lr = optimizer.lr
        self.warmup_steps = warmup_steps
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step < self.warmup_steps:
            self.optimizer.lr = self.base_lr * self.current_step / max(self.warmup_steps, 1)
        else:
            self.optimizer.lr = self.base_lr

    def get_lr(self) -> float:
        return self.optimizer.lr
