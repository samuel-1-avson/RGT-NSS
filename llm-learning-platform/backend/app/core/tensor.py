"""
Tensor Engine — PyTorch-backed Educational Tensor Operations

Wraps torch.Tensor with an educational API that exposes intermediate
computations, gradient information, and step-by-step breakdowns for
the interactive learning platform.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

from app.core.device import get_device, to_device


class Tensor:
    """
    Educational tensor wrapper around torch.Tensor.

    Provides the same API surface as the numpy version but backed by
    real PyTorch autograd for GPU-accelerated gradient computation.
    """

    def __init__(
        self,
        data: Union[np.ndarray, list, torch.Tensor, float],
        requires_grad: bool = False,
        name: str = "",
        _raw: bool = False,
    ):
        if _raw and isinstance(data, torch.Tensor):
            # Internal: keep tensor as-is (preserves computation graph)
            self._tensor = data
        elif isinstance(data, torch.Tensor):
            if data.requires_grad:
                self._tensor = data.float()
            else:
                self._tensor = data.detach().float()
        elif isinstance(data, np.ndarray):
            self._tensor = torch.from_numpy(data.astype(np.float32))
        elif isinstance(data, (list, tuple)):
            self._tensor = torch.tensor(data, dtype=torch.float32)
        else:
            self._tensor = torch.tensor(float(data), dtype=torch.float32)

        if not _raw:
            self._tensor = to_device(self._tensor)
        if requires_grad and not self._tensor.requires_grad:
            self._tensor = self._tensor.requires_grad_(True)

        self.name = name
        self._grad: Optional[np.ndarray] = None

    @staticmethod
    def _wrap(t: torch.Tensor) -> Tensor:
        """Wrap a torch tensor preserving the computation graph."""
        return Tensor(t, _raw=True)

    # ─── Properties ──────────────────────────────────────────

    @property
    def data(self) -> np.ndarray:
        """Return data as numpy array (for API compatibility)."""
        return self._tensor.detach().cpu().numpy()

    @data.setter
    def data(self, value: np.ndarray):
        self._tensor = to_device(torch.from_numpy(value.astype(np.float32)))
        if self._tensor.requires_grad:
            self._tensor = self._tensor.requires_grad_(True)

    @property
    def torch_tensor(self) -> torch.Tensor:
        """Direct access to the underlying PyTorch tensor."""
        return self._tensor

    @property
    def shape(self) -> tuple:
        return tuple(self._tensor.shape)

    @property
    def ndim(self) -> int:
        return self._tensor.ndim

    @property
    def dtype(self) -> str:
        return str(self._tensor.dtype)

    @property
    def device(self) -> str:
        return str(self._tensor.device)

    @property
    def T(self) -> Tensor:
        """Transpose (2D only)."""
        return Tensor(self._tensor.T)

    def numel(self) -> int:
        return self._tensor.numel()

    def size(self, dim: Optional[int] = None):
        if dim is not None:
            return self._tensor.size(dim)
        return tuple(self._tensor.shape)

    @property
    def requires_grad(self) -> bool:
        return self._tensor.requires_grad

    @property
    def grad(self) -> Optional[np.ndarray]:
        if self._tensor.grad is not None:
            return self._tensor.grad.detach().cpu().numpy()
        return self._grad

    @grad.setter
    def grad(self, value):
        if value is None:
            self._grad = None
            if self._tensor.grad is not None:
                self._tensor.grad = None
        elif isinstance(value, np.ndarray):
            self._grad = value

    # ─── Arithmetic ──────────────────────────────────────────

    def __add__(self, other: Union[Tensor, float]) -> Tensor:
        other_t = other._tensor if isinstance(other, Tensor) else other
        return Tensor._wrap(self._tensor + other_t)

    def __radd__(self, other) -> Tensor:
        return Tensor._wrap(other + self._tensor)

    def __mul__(self, other: Union[Tensor, float]) -> Tensor:
        other_t = other._tensor if isinstance(other, Tensor) else other
        return Tensor._wrap(self._tensor * other_t)

    def __rmul__(self, other) -> Tensor:
        return Tensor._wrap(other * self._tensor)

    def __sub__(self, other: Union[Tensor, float]) -> Tensor:
        other_t = other._tensor if isinstance(other, Tensor) else other
        return Tensor._wrap(self._tensor - other_t)

    def __rsub__(self, other) -> Tensor:
        return Tensor._wrap(other - self._tensor)

    def __truediv__(self, other: Union[Tensor, float]) -> Tensor:
        other_t = other._tensor if isinstance(other, Tensor) else other
        return Tensor._wrap(self._tensor / other_t)

    def __pow__(self, exponent) -> Tensor:
        return Tensor._wrap(self._tensor ** exponent)

    def __neg__(self) -> Tensor:
        return Tensor._wrap(-self._tensor)

    def __matmul__(self, other: Tensor) -> Tensor:
        return Tensor._wrap(self._tensor @ other._tensor)

    def __getitem__(self, key) -> Tensor:
        return Tensor._wrap(self._tensor[key])

    def __len__(self) -> int:
        return len(self._tensor)

    def __repr__(self) -> str:
        name_str = f" name='{self.name}'" if self.name else ""
        return f"Tensor(shape={list(self.shape)}, device={self.device}{name_str})"

    # ─── Reductions ──────────────────────────────────────────

    def sum(self, axis=None, keepdims=False) -> Tensor:
        if axis is None:
            return Tensor._wrap(self._tensor.sum())
        return Tensor._wrap(self._tensor.sum(dim=axis, keepdim=keepdims))

    def mean(self, axis=None, keepdims=False) -> Tensor:
        if axis is None:
            return Tensor._wrap(self._tensor.mean())
        return Tensor._wrap(self._tensor.mean(dim=axis, keepdim=keepdims))

    def max(self, axis=None) -> Tensor:
        if axis is None:
            return Tensor._wrap(self._tensor.max())
        return Tensor._wrap(self._tensor.max(dim=axis).values)

    def min(self, axis=None) -> Tensor:
        if axis is None:
            return Tensor._wrap(self._tensor.min())
        return Tensor._wrap(self._tensor.min(dim=axis).values)

    # ─── Activations ─────────────────────────────────────────

    def relu(self) -> Tensor:
        return Tensor._wrap(F.relu(self._tensor))

    def sigmoid(self) -> Tensor:
        return Tensor._wrap(torch.sigmoid(self._tensor))

    def tanh(self) -> Tensor:
        return Tensor._wrap(torch.tanh(self._tensor))

    def gelu(self) -> Tensor:
        return Tensor._wrap(F.gelu(self._tensor))

    def silu(self) -> Tensor:
        return Tensor._wrap(F.silu(self._tensor))

    def softmax(self, axis: int = -1) -> Tensor:
        return Tensor._wrap(F.softmax(self._tensor, dim=axis))

    def log_softmax(self, axis: int = -1) -> Tensor:
        return Tensor._wrap(F.log_softmax(self._tensor, dim=axis))

    # ─── Shape Operations ────────────────────────────────────

    def reshape(self, *shape) -> Tensor:
        return Tensor(self._tensor.reshape(*shape))

    def transpose(self, dim0: int, dim1: int) -> Tensor:
        return Tensor(self._tensor.transpose(dim0, dim1))

    def unsqueeze(self, dim: int) -> Tensor:
        return Tensor(self._tensor.unsqueeze(dim))

    def squeeze(self, dim: Optional[int] = None) -> Tensor:
        if dim is None:
            return Tensor(self._tensor.squeeze())
        return Tensor(self._tensor.squeeze(dim))

    # ─── Loss Functions ──────────────────────────────────────

    def cross_entropy(self, targets: Union[np.ndarray, torch.Tensor]) -> Tensor:
        """Compute cross-entropy loss. self = logits (B, seq, vocab), targets = (B, seq)."""
        if isinstance(targets, np.ndarray):
            targets = to_device(torch.from_numpy(targets).long())
        elif isinstance(targets, torch.Tensor):
            targets = to_device(targets.long())

        logits = self._tensor
        if logits.ndim == 3:
            B, S, V = logits.shape
            logits = logits.reshape(B * S, V)
            targets = targets.reshape(B * S)

        loss = F.cross_entropy(logits, targets)
        return Tensor(loss)

    def mse_loss(self, target: Tensor) -> Tensor:
        return Tensor(F.mse_loss(self._tensor, target._tensor))

    # ─── Autograd ────────────────────────────────────────────

    def backward(self):
        """Run backpropagation through the computation graph."""
        self._tensor.backward()

    def zero_grad(self):
        """Zero out gradients."""
        if self._tensor.grad is not None:
            self._tensor.grad.zero_()
        self._grad = None

    # ─── Utility ─────────────────────────────────────────────

    def numpy(self) -> np.ndarray:
        return self.data

    def item(self) -> float:
        return self._tensor.item()

    def clone(self) -> Tensor:
        return Tensor(self._tensor.clone(), requires_grad=self.requires_grad, name=self.name)

    def detach(self) -> Tensor:
        return Tensor(self._tensor.detach(), name=self.name)

    @staticmethod
    def zeros(*shape, requires_grad=False) -> Tensor:
        return Tensor(torch.zeros(*shape), requires_grad=requires_grad)

    @staticmethod
    def ones(*shape, requires_grad=False) -> Tensor:
        return Tensor(torch.ones(*shape), requires_grad=requires_grad)

    @staticmethod
    def randn(*shape, requires_grad=False) -> Tensor:
        return Tensor(torch.randn(*shape), requires_grad=requires_grad)

    @staticmethod
    def from_numpy(arr: np.ndarray, requires_grad=False) -> Tensor:
        return Tensor(arr, requires_grad=requires_grad)
