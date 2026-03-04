"""
Training Engine

Manages the full training loop, metrics collection, gradient clipping,
and real-time progress broadcasting for visualization.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional

import numpy as np

from app.core.model import MicroGPT, GPTConfig
from app.core.optimizers import AdamW, CosineScheduler
from app.core.tensor import Tensor


class TrainingStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TrainingMetrics:
    step: int = 0
    epoch: int = 0
    loss: float = 0.0
    learning_rate: float = 0.0
    grad_norm: float = 0.0
    tokens_per_sec: float = 0.0
    elapsed_seconds: float = 0.0
    perplexity: float = 0.0


@dataclass
class TrainingConfig:
    num_epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_steps: int = 1000
    grad_clip: float = 1.0
    eval_interval: int = 50
    checkpoint_interval: int = 200
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = False


@dataclass
class TrainingSession:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: TrainingStatus = TrainingStatus.IDLE
    config: Optional[TrainingConfig] = None
    model_config: Optional[GPTConfig] = None
    metrics_history: List[TrainingMetrics] = field(default_factory=list)
    best_loss: float = float("inf")
    current_step: int = 0
    total_steps: int = 0


class TrainingEngine:
    """
    Manages model training with real-time metrics and visualization support.
    """

    def __init__(self):
        self.sessions: Dict[str, TrainingSession] = {}

    def create_session(
        self,
        model_config: GPTConfig,
        training_config: TrainingConfig,
    ) -> TrainingSession:
        session = TrainingSession(
            config=training_config,
            model_config=model_config,
            total_steps=training_config.max_steps,
        )
        self.sessions[session.session_id] = session
        return session

    def train(
        self,
        session_id: str,
        data: np.ndarray,
        on_step: Optional[Callable[[TrainingMetrics], None]] = None,
    ) -> TrainingSession:
        """
        Execute training loop.

        Args:
            session_id: session identifier
            data: (num_samples, seq_len) training data
            on_step: callback invoked after each step with metrics
        Returns:
            Updated TrainingSession
        """
        session = self.sessions[session_id]
        session.status = TrainingStatus.RUNNING
        config = session.config

        # Build model
        model = MicroGPT(session.model_config)
        model.set_training(True)

        # Optimizer
        optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Scheduler
        scheduler = CosineScheduler(
            optimizer,
            total_steps=config.max_steps,
            warmup_steps=config.warmup_steps,
        )

        num_samples = len(data)
        start_time = time.time()
        accum_steps = max(config.gradient_accumulation_steps, 1)
        loss_scale = 65536.0 if config.mixed_precision else 1.0

        try:
            for step in range(config.max_steps):
                if session.status == TrainingStatus.PAUSED:
                    break

                accumulated_loss = 0.0
                model.zero_grad()

                for accum_step in range(accum_steps):
                    # Sample batch
                    indices = np.random.randint(0, num_samples, size=config.batch_size)
                    batch = data[indices]
                    inputs = batch[:, :-1]
                    targets = batch[:, 1:]

                    # Forward pass with optional FP16-style reduced precision
                    result = model.forward(inputs, targets)
                    loss = result["loss"]

                    if config.mixed_precision:
                        # Approximate mixed precision by quantizing loss to float16
                        loss = float(np.float16(loss))

                    accumulated_loss += loss / accum_steps

                    # Backward pass with loss scaling
                    loss_tensor = result["loss_tensor"]
                    loss_tensor.data = loss_tensor.data * (loss_scale / accum_steps)
                    loss_tensor.backward()

                # Unscale gradients if using mixed precision
                if config.mixed_precision:
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad = p.grad / loss_scale

                # Gradient clipping
                grad_norm = self._clip_gradients(model.parameters(), config.grad_clip)

                # Optimizer step
                optimizer.step()
                scheduler.step()

                elapsed = time.time() - start_time
                effective_batch = config.batch_size * accum_steps
                tokens_per_sec = (
                    (step + 1) * effective_batch * inputs.shape[1] / max(elapsed, 1e-6)
                )

                metrics = TrainingMetrics(
                    step=step,
                    epoch=step * effective_batch // num_samples,
                    loss=accumulated_loss,
                    learning_rate=optimizer.lr,
                    grad_norm=grad_norm,
                    tokens_per_sec=tokens_per_sec,
                    elapsed_seconds=elapsed,
                    perplexity=min(float(np.exp(accumulated_loss)), 1e6),
                )

                session.metrics_history.append(metrics)
                session.current_step = step

                if loss < session.best_loss:
                    session.best_loss = loss

                if on_step:
                    on_step(metrics)

            session.status = TrainingStatus.COMPLETED

        except Exception as e:
            session.status = TrainingStatus.FAILED
            raise

        return session

    def pause_session(self, session_id: str):
        if session_id in self.sessions:
            self.sessions[session_id].status = TrainingStatus.PAUSED

    def get_session(self, session_id: str) -> Optional[TrainingSession]:
        return self.sessions.get(session_id)

    @staticmethod
    def _clip_gradients(params: List[Tensor], max_norm: float) -> float:
        """Clip gradients by global norm. Returns the original norm."""
        total_norm_sq = 0.0
        for p in params:
            if p.grad is not None:
                total_norm_sq += float(np.sum(p.grad ** 2))
        total_norm = float(np.sqrt(total_norm_sq))

        if total_norm > max_norm and total_norm > 0:
            scale = max_norm / total_norm
            for p in params:
                if p.grad is not None:
                    p.grad *= scale

        return total_norm

    # Real educational text corpus for training data
    TRAINING_CORPUS = [
        "The transformer architecture was introduced in the paper Attention Is All You Need in 2017.",
        "Language models predict the next token in a sequence based on the context of preceding tokens.",
        "Self-attention allows each position in the sequence to attend to all other positions.",
        "The key, query, and value projections are fundamental components of the attention mechanism.",
        "Positional encoding provides information about the relative or absolute position of tokens.",
        "The feed-forward network in each transformer layer applies two linear transformations.",
        "Layer normalization stabilizes training by normalizing activations across the feature dimension.",
        "Dropout is a regularization technique that randomly sets a fraction of inputs to zero.",
        "The softmax function converts logits into a probability distribution over possible tokens.",
        "Cross-entropy loss measures the difference between predicted and actual token distributions.",
        "Gradient descent optimizes model parameters by following the negative gradient of the loss.",
        "Learning rate warmup gradually increases the learning rate at the beginning of training.",
        "Weight decay adds a penalty term to prevent model weights from growing too large.",
        "Residual connections allow gradients to flow directly through the network without degradation.",
        "Multi-head attention projects queries, keys, and values into multiple subspaces.",
        "The vocabulary size determines the number of unique tokens the model can process.",
        "Tokenization splits text into smaller units that the model can understand and process.",
        "Embedding layers convert discrete token indices into continuous vector representations.",
        "The model dimension determines the size of the hidden representations throughout the network.",
        "Beam search explores multiple generation paths to find higher probability sequences.",
        "Temperature scaling controls the randomness of the sampling distribution during generation.",
        "Top-k sampling restricts generation to the k most probable next tokens at each step.",
        "Nucleus sampling selects from the smallest set of tokens whose cumulative probability exceeds p.",
        "Perplexity measures how well a probability model predicts a sample of text data.",
        "BLEU score evaluates the quality of machine-generated text against reference translations.",
        "ROUGE metrics assess the quality of summaries by comparing overlap with reference summaries.",
        "Fine-tuning adapts a pretrained model to a specific downstream task using labeled data.",
        "Transfer learning leverages knowledge learned on one task to improve performance on another.",
        "Data augmentation increases training set diversity by applying transformations to existing data.",
        "Overfitting occurs when a model learns training data noise rather than underlying patterns.",
        "Regularization techniques help prevent overfitting by constraining the model complexity.",
        "The attention mask prevents the model from attending to future tokens during training.",
        "Causal language modeling trains the model to predict each token given only previous tokens.",
        "The encoder processes input sequences while the decoder generates output sequences.",
        "Autoregressive generation produces tokens one at a time conditioned on previous outputs.",
        "The context window determines the maximum number of tokens the model can process at once.",
        "Gradient clipping prevents exploding gradients by limiting the gradient magnitude.",
        "Mixed precision training uses both float16 and float32 to reduce memory and increase speed.",
        "Reinforcement learning from human feedback aligns model outputs with human preferences.",
        "Low-rank adaptation reduces the number of trainable parameters by decomposing weight updates.",
    ]

    @staticmethod
    def generate_sample_data(
        vocab_size: int = 256,
        num_samples: int = 1000,
        seq_len: int = 64,
    ) -> np.ndarray:
        """Generate training data by encoding real English text into token sequences."""
        corpus_text = " ".join(TrainingEngine.TRAINING_CORPUS)
        encoded = np.array([ord(c) % vocab_size for c in corpus_text], dtype=np.int64)
        corpus_len = len(encoded)

        data = np.zeros((num_samples, seq_len), dtype=np.int64)
        for i in range(num_samples):
            start = (i * 37) % max(corpus_len - seq_len, 1)
            for j in range(seq_len):
                data[i, j] = encoded[(start + j) % corpus_len]

        return data
