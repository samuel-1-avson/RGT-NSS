"""
Training engine for GPT models with real-time metrics and checkpointing.
"""

import numpy as np
import time
from typing import Optional, Callable, List, Dict, Any, Generator
from dataclasses import dataclass, field
from datetime import datetime

from app.models.gpt import MicroGPT, GPTConfig
from app.core.optimizer import AdamW, clip_gradients, LinearWarmupCosineDecay


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Model
    model_config: GPTConfig = field(default_factory=GPTConfig)
    
    # Training
    batch_size: int = 32
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 100
    max_steps: int = 10000
    max_epochs: Optional[int] = None
    
    # Optimization
    grad_clip: float = 1.0
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.999)
    
    # Evaluation
    eval_interval: int = 100
    eval_steps: int = 10
    
    # Checkpointing
    checkpoint_interval: int = 1000
    checkpoint_dir: str = "./checkpoints"
    
    # Logging
    log_interval: int = 10
    
    # Data
    seq_length: int = 256


@dataclass
class TrainingMetrics:
    """Metrics from a training step."""
    step: int
    epoch: int
    loss: float
    perplexity: float
    learning_rate: float
    grad_norm: float
    tokens_per_sec: float
    time_elapsed: float
    time_remaining: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'step': self.step,
            'epoch': self.epoch,
            'loss': round(self.loss, 4),
            'perplexity': round(self.perplexity, 2),
            'learning_rate': round(self.learning_rate, 6),
            'grad_norm': round(self.grad_norm, 4),
            'tokens_per_sec': round(self.tokens_per_sec, 1),
            'time_elapsed': round(self.time_elapsed, 1),
            'time_remaining': round(self.time_remaining, 1) if self.time_remaining else None,
        }


class DataLoader:
    """
    Simple data loader for text data.
    Creates batches of sequences for language modeling.
    """
    
    def __init__(self, data: np.ndarray, batch_size: int, seq_length: int,
                 shuffle: bool = True):
        """
        Args:
            data: Flat array of token IDs
            batch_size: Number of sequences per batch
            seq_length: Length of each sequence
            shuffle: Whether to shuffle data
        """
        self.data = data
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.shuffle = shuffle
        
        # Calculate number of batches
        self.num_tokens = len(data)
        self.num_batches = self.num_tokens // (batch_size * seq_length)
        
    def __iter__(self):
        """Iterate over batches."""
        if self.shuffle:
            # Shuffle data
            indices = np.random.permutation(self.num_tokens - self.seq_length - 1)
        else:
            indices = np.arange(self.num_tokens - self.seq_length - 1)
        
        batch_idx = 0
        while batch_idx < self.num_batches:
            # Get batch indices
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(indices))
            batch_indices = indices[start_idx:end_idx]
            
            # Create sequences
            x = np.zeros((len(batch_indices), self.seq_length), dtype=np.int32)
            y = np.zeros((len(batch_indices), self.seq_length), dtype=np.int32)
            
            for i, idx in enumerate(batch_indices):
                x[i] = self.data[idx:idx + self.seq_length]
                y[i] = self.data[idx + 1:idx + self.seq_length + 1]
            
            yield x, y
            batch_idx += 1
    
    def __len__(self):
        return self.num_batches


class TrainingEngine:
    """
    Training engine for GPT models.
    
    Handles:
    - Training loop with gradient accumulation
    - Learning rate scheduling
    - Gradient clipping
    - Checkpointing
    - Real-time metrics reporting
    """
    
    def __init__(self, model: MicroGPT, config: TrainingConfig):
        self.model = model
        self.config = config
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=config.betas,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = LinearWarmupCosineDecay(
            self.optimizer,
            warmup_steps=config.warmup_steps,
            total_steps=config.max_steps,
            min_lr_ratio=config.min_learning_rate / config.learning_rate
        )
        
        # State
        self.current_step = 0
        self.current_epoch = 0
        self.is_training = False
        self.should_stop = False
        
        # Metrics history
        self.metrics_history: List[TrainingMetrics] = []
        self.best_loss = float('inf')
        
        # Callbacks
        self.callbacks: List[Callable] = []
        
        # Timing
        self.start_time = None
        self.step_times: List[float] = []
    
    def add_callback(self, callback: Callable):
        """Add a callback for metrics reporting."""
        self.callbacks.append(callback)
    
    def train_step(self, x: np.ndarray, y: np.ndarray) -> TrainingMetrics:
        """
        Execute single training step.
        
        Args:
            x: Input tokens (batch_size, seq_length)
            y: Target tokens (batch_size, seq_length)
        
        Returns:
            TrainingMetrics for this step
        """
        step_start = time.time()
        
        # Forward pass
        logits, loss, _ = self.model.forward(x, targets=y)
        
        # Backward pass
        self.model.zero_grad()
        loss.backward()
        
        # Gradient clipping
        grad_norm = clip_gradients(self.model.parameters(), self.config.grad_clip)
        
        # Optimizer step
        self.optimizer.step()
        
        # Update learning rate
        current_lr = self.scheduler.step()
        
        # Calculate metrics
        loss_val = loss.item()
        perplexity = np.exp(loss_val)
        
        # Timing
        step_time = time.time() - step_start
        self.step_times.append(step_time)
        if len(self.step_times) > 100:
            self.step_times.pop(0)
        
        # Tokens per second
        tokens_per_sec = (x.shape[0] * x.shape[1]) / step_time
        
        # Time estimates
        time_elapsed = time.time() - self.start_time if self.start_time else 0
        avg_step_time = np.mean(self.step_times)
        steps_remaining = self.config.max_steps - self.current_step
        time_remaining = avg_step_time * steps_remaining if steps_remaining > 0 else None
        
        metrics = TrainingMetrics(
            step=self.current_step,
            epoch=self.current_epoch,
            loss=loss_val,
            perplexity=perplexity,
            learning_rate=current_lr,
            grad_norm=grad_norm,
            tokens_per_sec=tokens_per_sec,
            time_elapsed=time_elapsed,
            time_remaining=time_remaining
        )
        
        self.metrics_history.append(metrics)
        
        # Update best loss
        if loss_val < self.best_loss:
            self.best_loss = loss_val
        
        # Notify callbacks
        for callback in self.callbacks:
            callback(metrics)
        
        return metrics
    
    def train_epoch(self, train_loader: DataLoader, 
                    val_loader: Optional[DataLoader] = None) -> Generator[TrainingMetrics, None, None]:
        """
        Train for one epoch.
        
        Yields:
            TrainingMetrics after each step
        """
        self.model.train()
        
        for x, y in train_loader:
            if self.should_stop or self.current_step >= self.config.max_steps:
                break
            
            self.current_step += 1
            metrics = self.train_step(x, y)
            
            yield metrics
            
            # Evaluation
            if val_loader and self.current_step % self.config.eval_interval == 0:
                val_loss = self.evaluate(val_loader)
                print(f"Step {self.current_step}: Val Loss = {val_loss:.4f}")
            
            # Checkpointing
            if self.current_step % self.config.checkpoint_interval == 0:
                self.save_checkpoint()
    
    def evaluate(self, val_loader: DataLoader, max_steps: Optional[int] = None) -> float:
        """
        Evaluate model on validation set.
        
        Returns:
            Average loss
        """
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        for i, (x, y) in enumerate(val_loader):
            if max_steps and i >= max_steps:
                break
            
            logits, loss, _ = self.model.forward(x, targets=y)
            total_loss += loss.item()
            num_batches += 1
        
        self.model.train()
        
        return total_loss / num_batches if num_batches > 0 else float('inf')
    
    def train(self, train_data: np.ndarray, val_data: Optional[np.ndarray] = None) -> Generator[TrainingMetrics, None, None]:
        """
        Full training loop.
        
        Args:
            train_data: Training token IDs
            val_data: Optional validation token IDs
        
        Yields:
            TrainingMetrics after each step
        """
        self.is_training = True
        self.should_stop = False
        self.start_time = time.time()
        
        train_loader = DataLoader(
            train_data,
            batch_size=self.config.batch_size,
            seq_length=self.config.seq_length,
            shuffle=True
        )
        
        val_loader = None
        if val_data is not None:
            val_loader = DataLoader(
                val_data,
                batch_size=self.config.batch_size,
                seq_length=self.config.seq_length,
                shuffle=False
            )
        
        epoch = 0
        while self.current_step < self.config.max_steps:
            if self.should_stop:
                break
            
            self.current_epoch = epoch
            print(f"\nEpoch {epoch + 1}")
            
            for metrics in self.train_epoch(train_loader, val_loader):
                yield metrics
            
            epoch += 1
            
            if self.config.max_epochs and epoch >= self.config.max_epochs:
                break
        
        self.is_training = False
        print(f"\nTraining complete! Best loss: {self.best_loss:.4f}")
    
    def stop(self):
        """Stop training gracefully."""
        self.should_stop = True
        self.is_training = False
    
    def save_checkpoint(self, path: Optional[str] = None):
        """Save model checkpoint."""
        import os
        import pickle
        
        if path is None:
            os.makedirs(self.config.checkpoint_dir, exist_ok=True)
            path = os.path.join(
                self.config.checkpoint_dir,
                f"checkpoint_step_{self.current_step}.pkl"
            )
        
        checkpoint = {
            'model_state': self.model.state_dict(),
            'optimizer_state': {
                'm': self.optimizer.m,
                'v': self.optimizer.v,
                'step_count': self.optimizer.step_count,
            },
            'scheduler_state': {
                'step_count': self.scheduler.optimizer.step_count,
            },
            'current_step': self.current_step,
            'current_epoch': self.current_epoch,
            'best_loss': self.best_loss,
            'metrics_history': self.metrics_history,
            'config': self.config,
        }
        
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        import pickle
        
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.m = checkpoint['optimizer_state']['m']
        self.optimizer.v = checkpoint['optimizer_state']['v']
        self.optimizer.step_count = checkpoint['optimizer_state']['step_count']
        self.scheduler.optimizer.step_count = checkpoint['scheduler_state']['step_count']
        self.current_step = checkpoint['current_step']
        self.current_epoch = checkpoint['current_epoch']
        self.best_loss = checkpoint['best_loss']
        self.metrics_history = checkpoint['metrics_history']
        
        print(f"Checkpoint loaded from {path}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current training status."""
        return {
            'is_training': self.is_training,
            'current_step': self.current_step,
            'current_epoch': self.current_epoch,
            'best_loss': self.best_loss,
            'progress': self.current_step / self.config.max_steps,
        }


class Callback:
    """Base callback class."""
    
    def on_train_begin(self, engine: TrainingEngine):
        pass
    
    def on_train_end(self, engine: TrainingEngine):
        pass
    
    def on_step_end(self, metrics: TrainingMetrics):
        pass
    
    def on_epoch_end(self, epoch: int, metrics: TrainingMetrics):
        pass


class MetricsLogger(Callback):
    """Callback to log metrics to file."""
    
    def __init__(self, log_file: str):
        self.log_file = log_file
    
    def on_step_end(self, metrics: TrainingMetrics):
        with open(self.log_file, 'a') as f:
            f.write(f"{metrics.to_dict()}\n")


class EarlyStopping(Callback):
    """Early stopping callback."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
    
    def on_step_end(self, metrics: TrainingMetrics):
        if metrics.loss < self.best_loss - self.min_delta:
            self.best_loss = metrics.loss
            self.counter = 0
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            print(f"Early stopping triggered after {self.patience} steps without improvement")
