"""
Optimizer Engine — PyTorch Optimizer Wrappers with Educational API

Wraps torch.optim optimizers with step-by-step visualization
and comparison capabilities for the learning platform.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, StepLR, ExponentialLR, OneCycleLR,
    LinearLR, CosineAnnealingWarmRestarts,
)

from app.core.device import get_device


@dataclass
class OptimizerConfig:
    """Configuration for optimizer comparison."""
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    momentum: float = 0.9
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8


def create_optimizer(
    name: str, parameters, config: Optional[OptimizerConfig] = None
) -> torch.optim.Optimizer:
    """Create a PyTorch optimizer by name."""
    cfg = config or OptimizerConfig()
    name = name.lower()

    if name == "sgd":
        return optim.SGD(parameters, lr=cfg.learning_rate, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    elif name == "adam":
        return optim.Adam(parameters, lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2), eps=cfg.epsilon)
    elif name == "adamw":
        return optim.AdamW(parameters, lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2), eps=cfg.epsilon, weight_decay=cfg.weight_decay)
    elif name == "rmsprop":
        return optim.RMSprop(parameters, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    elif name == "adagrad":
        return optim.Adagrad(parameters, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def compare_optimizers(
    optimizer_names: Optional[List[str]] = None,
    num_steps: int = 100,
    d_model: int = 64,
    learning_rate: float = 1e-3,
) -> Dict:
    """
    Compare optimizers by training a small model with each.
    Uses real PyTorch optimization on GPU.
    """
    if optimizer_names is None:
        optimizer_names = ["sgd", "adam", "adamw", "rmsprop"]

    device = get_device()
    results = {}

    for opt_name in optimizer_names:
        # Create a simple model for fair comparison
        model = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
        ).to(device)

        config = OptimizerConfig(learning_rate=learning_rate)
        optimizer = create_optimizer(opt_name, model.parameters(), config)

        losses = []
        torch.manual_seed(42)
        x = torch.randn(8, d_model, device=device)
        target = torch.randn(8, d_model, device=device)

        for step in range(num_steps):
            optimizer.zero_grad()
            output = model(x)
            loss = nn.functional.mse_loss(output, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        results[opt_name] = {
            "final_loss": round(losses[-1], 6),
            "best_loss": round(min(losses), 6),
            "convergence_step": next((i for i, l in enumerate(losses) if l < losses[0] * 0.1), num_steps),
            "loss_curve": [round(l, 6) for l in losses[::max(1, num_steps // 20)]],
        }

    return {
        "optimizers": results,
        "num_steps": num_steps,
        "best": min(results, key=lambda k: results[k]["final_loss"]),
        "device": str(device),
    }


def analyze_learning_rates(
    lr_values: Optional[List[float]] = None,
    num_steps: int = 100,
    d_model: int = 64,
) -> Dict:
    """Compare different learning rates on a simple optimization task."""
    if lr_values is None:
        lr_values = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

    device = get_device()
    results = {}

    for lr in lr_values:
        model = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
        ).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=lr)

        losses = []
        torch.manual_seed(42)
        x = torch.randn(8, d_model, device=device)
        target = torch.randn(8, d_model, device=device)

        for step in range(num_steps):
            optimizer.zero_grad()
            output = model(x)
            loss = nn.functional.mse_loss(output, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        label = f"lr_{lr}"
        results[label] = {
            "learning_rate": lr,
            "final_loss": round(losses[-1], 6),
            "best_loss": round(min(losses), 6),
            "diverged": losses[-1] > losses[0] * 2,
            "loss_curve": [round(l, 6) for l in losses[::max(1, num_steps // 20)]],
        }

    return {
        "results": results,
        "num_steps": num_steps,
        "device": str(device),
    }


def compare_schedulers(
    num_steps: int = 200,
    learning_rate: float = 1e-3,
) -> Dict:
    """Compare different LR schedulers."""
    scheduler_configs = {
        "constant": lambda opt: None,
        "cosine": lambda opt: CosineAnnealingLR(opt, T_max=num_steps),
        "step_decay": lambda opt: StepLR(opt, step_size=50, gamma=0.5),
        "exponential": lambda opt: ExponentialLR(opt, gamma=0.99),
        "cosine_restarts": lambda opt: CosineAnnealingWarmRestarts(opt, T_0=50),
    }

    results = {}
    for name, sched_fn in scheduler_configs.items():
        lr_history = []
        dummy_param = torch.randn(1, requires_grad=True)
        optimizer = optim.AdamW([dummy_param], lr=learning_rate)
        scheduler = sched_fn(optimizer)

        for step in range(num_steps):
            lr_history.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            if scheduler:
                scheduler.step()

        results[name] = {
            "lr_schedule": [round(lr, 8) for lr in lr_history[::max(1, num_steps // 40)]],
            "final_lr": round(lr_history[-1], 8),
            "min_lr": round(min(lr_history), 8),
        }

    return {
        "schedulers": results,
        "num_steps": num_steps,
        "initial_lr": learning_rate,
    }
