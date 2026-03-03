"""
Optimization algorithms for training neural networks.
"""

import numpy as np
from typing import List, Tuple
from .tensor import Tensor


class Optimizer:
    """Base optimizer class."""
    
    def __init__(self, parameters: List[Tensor], lr: float = 0.001):
        self.parameters = [p for p in parameters if p.requires_grad]
        self.lr = lr
        self.step_count = 0
    
    def zero_grad(self):
        """Zero all parameter gradients."""
        for p in self.parameters:
            p.zero_grad()
    
    def step(self):
        """Perform optimization step."""
        raise NotImplementedError
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.lr
    
    def set_lr(self, lr: float):
        """Set learning rate."""
        self.lr = lr


class SGD(Optimizer):
    """
    Stochastic Gradient Descent with optional momentum.
    
    v_t = momentum * v_{t-1} + grad
    p_t = p_{t-1} - lr * v_t
    """
    
    def __init__(self, parameters: List[Tensor], lr: float = 0.01,
                 momentum: float = 0.0, weight_decay: float = 0.0):
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        
        # Velocity for momentum
        self.velocities = [np.zeros_like(p.data) for p in self.parameters]
    
    def step(self):
        self.step_count += 1
        
        for i, p in enumerate(self.parameters):
            grad = p.grad
            
            # Weight decay (L2 regularization)
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * p.data
            
            # Momentum
            if self.momentum != 0:
                self.velocities[i] = self.momentum * self.velocities[i] + grad
                grad = self.velocities[i]
            
            # Update parameters
            p.data -= self.lr * grad


class Adam(Optimizer):
    """
    Adam optimizer (Kingma & Ba, 2015).
    
    Combines momentum (first moment) and RMSprop (second moment) with bias correction.
    
    m_t = β1 * m_{t-1} + (1 - β1) * grad
    v_t = β2 * v_{t-1} + (1 - β2) * grad^2
    m̂_t = m_t / (1 - β1^t)  # Bias correction
    v̂_t = v_t / (1 - β2^t)
    p_t = p_{t-1} - lr * m̂_t / (sqrt(v̂_t) + ε)
    """
    
    def __init__(self, parameters: List[Tensor], lr: float = 0.001,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0):
        super().__init__(parameters, lr)
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        # First and second moment estimates
        self.m = [np.zeros_like(p.data) for p in self.parameters]
        self.v = [np.zeros_like(p.data) for p in self.parameters]
    
    def step(self):
        self.step_count += 1
        beta1, beta2 = self.betas
        
        for i, p in enumerate(self.parameters):
            grad = p.grad
            
            # Weight decay (decoupled)
            if self.weight_decay != 0:
                p.data -= self.lr * self.weight_decay * p.data
            
            # Update biased first moment estimate
            self.m[i] = beta1 * self.m[i] + (1 - beta1) * grad
            
            # Update biased second raw moment estimate
            self.v[i] = beta2 * self.v[i] + (1 - beta2) * (grad ** 2)
            
            # Bias correction
            m_hat = self.m[i] / (1 - beta1 ** self.step_count)
            v_hat = self.v[i] / (1 - beta2 ** self.step_count)
            
            # Update parameters
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class AdamW(Optimizer):
    """
    AdamW optimizer (Loshchilov & Hutter, 2019).
    
    Fixes weight decay in Adam by decoupling it from the gradient update.
    This typically leads to better generalization.
    """
    
    def __init__(self, parameters: List[Tensor], lr: float = 0.001,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.01):
        super().__init__(parameters, lr)
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        # First and second moment estimates
        self.m = [np.zeros_like(p.data) for p in self.parameters]
        self.v = [np.zeros_like(p.data) for p in self.parameters]
    
    def step(self):
        self.step_count += 1
        beta1, beta2 = self.betas
        
        for i, p in enumerate(self.parameters):
            grad = p.grad
            
            # Decoupled weight decay
            if self.weight_decay != 0:
                p.data *= (1 - self.lr * self.weight_decay)
            
            # Update biased first moment estimate
            self.m[i] = beta1 * self.m[i] + (1 - beta1) * grad
            
            # Update biased second raw moment estimate
            self.v[i] = beta2 * self.v[i] + (1 - beta2) * (grad ** 2)
            
            # Bias correction
            m_hat = self.m[i] / (1 - beta1 ** self.step_count)
            v_hat = self.v[i] / (1 - beta2 ** self.step_count)
            
            # Update parameters
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class CosineAnnealingLR:
    """
    Cosine annealing learning rate scheduler.
    
    lr_t = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * T_cur / T_i))
    """
    
    def __init__(self, optimizer: Optimizer, T_max: int, eta_min: float = 0,
                 warmup_steps: int = 0):
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.eta_max = optimizer.lr
        self.warmup_steps = warmup_steps
    
    def step(self):
        """Update learning rate."""
        self.optimizer.step_count += 1
        step = self.optimizer.step_count
        
        # Warmup phase
        if step < self.warmup_steps:
            lr = self.eta_max * step / self.warmup_steps
        else:
            # Cosine annealing
            progress = (step - self.warmup_steps) / (self.T_max - self.warmup_steps)
            lr = self.eta_min + 0.5 * (self.eta_max - self.eta_min) * \
                 (1 + np.cos(np.pi * progress))
        
        self.optimizer.set_lr(lr)
        return lr
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.get_lr()


class LinearWarmupCosineDecay:
    """
    Linear warmup followed by cosine decay.
    Popular in transformer training (e.g., GPT, BERT).
    """
    
    def __init__(self, optimizer: Optimizer, warmup_steps: int,
                 total_steps: int, min_lr_ratio: float = 0.1):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = optimizer.lr
        self.min_lr = optimizer.lr * min_lr_ratio
    
    def step(self):
        """Update learning rate."""
        self.optimizer.step_count += 1
        step = self.optimizer.step_count
        
        if step < self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * step / self.warmup_steps
        elif step >= self.total_steps:
            lr = self.min_lr
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * \
                 (1 + np.cos(np.pi * progress))
        
        self.optimizer.set_lr(lr)
        return lr
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.get_lr()


class ReduceLROnPlateau:
    """
    Reduce learning rate when a metric has stopped improving.
    """
    
    def __init__(self, optimizer: Optimizer, mode: str = 'min',
                 factor: float = 0.5, patience: int = 10,
                 min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        
        self.best = float('inf') if mode == 'min' else float('-inf')
        self.num_bad_epochs = 0
    
    def step(self, metric: float):
        """Update based on metric."""
        current = metric
        
        if self.mode == 'min':
            improved = current < self.best
        else:
            improved = current > self.best
        
        if improved:
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        
        if self.num_bad_epochs > self.patience:
            new_lr = max(self.optimizer.get_lr() * self.factor, self.min_lr)
            self.optimizer.set_lr(new_lr)
            self.num_bad_epochs = 0
            return new_lr
        
        return self.optimizer.get_lr()


def clip_gradients(parameters: List[Tensor], max_norm: float) -> float:
    """
    Clip gradients by global norm.
    
    Args:
        parameters: List of parameters
        max_norm: Maximum allowed norm
    
    Returns:
        Total gradient norm before clipping
    """
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            total_norm += np.sum(p.grad ** 2)
    total_norm = np.sqrt(total_norm)
    
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for p in parameters:
            if p.grad is not None:
                p.grad *= clip_coef
    
    return float(total_norm)
