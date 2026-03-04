"""
Tensor Computation Engine

Custom tensor class with automatic differentiation support.
Built from scratch for educational transparency — every operation
is visible and inspectable.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np


class Tensor:
    """
    Custom tensor with autograd support for educational purposes.

    Wraps a NumPy ndarray and tracks the computational graph
    so that gradients can be computed via backpropagation.
    """

    def __init__(
        self,
        data: Union[np.ndarray, list, float, int],
        requires_grad: bool = False,
        _children: Tuple["Tensor", ...] = (),
        _op: str = "",
    ):
        if isinstance(data, np.ndarray):
            self.data = data.astype(np.float32)
        else:
            self.data = np.array(data, dtype=np.float32)

        self.requires_grad = requires_grad
        self.grad: Optional[np.ndarray] = None
        self._backward = lambda: None  # no-op by default
        self._prev = set(_children)
        self._op = _op

        if requires_grad:
            self.grad = np.zeros_like(self.data)

    # ─── Properties ──────────────────────────────────────────
    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def T(self) -> "Tensor":
        return self.transpose()

    def size(self, dim: Optional[int] = None):
        if dim is None:
            return self.shape
        return self.shape[dim]

    def numel(self) -> int:
        return self.data.size

    # ─── Arithmetic Operations ───────────────────────────────
    def __add__(self, other: Union["Tensor", float, int]) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="+",
        )

        def _backward():
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad, self.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(out.grad, other.shape)

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __neg__(self) -> "Tensor":
        return self * (-1)

    def __sub__(self, other: Union["Tensor", float, int]) -> "Tensor":
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other: Union["Tensor", float, int]) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="*",
        )

        def _backward():
            if self.requires_grad:
                self.grad += _unbroadcast(other.data * out.grad, self.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(self.data * out.grad, other.shape)

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other: Union["Tensor", float, int]) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data / other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="/",
        )

        def _backward():
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad / other.data, self.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(
                    -self.data * out.grad / (other.data ** 2), other.shape
                )

        out._backward = _backward
        return out

    def __rtruediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other / self

    def __pow__(self, exp: Union[float, int]) -> "Tensor":
        out = Tensor(
            self.data ** exp,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op=f"**{exp}",
        )

        def _backward():
            if self.requires_grad:
                self.grad += exp * (self.data ** (exp - 1)) * out.grad

        out._backward = _backward
        return out

    # ─── Matrix Multiplication ───────────────────────────────
    def matmul(self, other: "Tensor") -> "Tensor":
        out = Tensor(
            self.data @ other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="@",
        )

        def _backward():
            if self.requires_grad:
                if self.data.ndim == 1:
                    self.grad += out.grad @ other.data.T
                elif self.data.ndim >= 2:
                    self.grad += out.grad @ _swap_last_two(other.data)
            if other.requires_grad:
                if other.data.ndim == 1:
                    other.grad += self.data.T @ out.grad
                elif other.data.ndim >= 2:
                    other.grad += _swap_last_two(self.data) @ out.grad

        out._backward = _backward
        return out

    def __matmul__(self, other: "Tensor") -> "Tensor":
        return self.matmul(other)

    # ─── Reduction Operations ────────────────────────────────
    def sum(self, axis: Optional[int] = None, keepdims: bool = False) -> "Tensor":
        out = Tensor(
            self.data.sum(axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="sum",
        )

        def _backward():
            if self.requires_grad:
                grad = out.grad
                if axis is not None and not keepdims:
                    grad = np.expand_dims(grad, axis=axis)
                self.grad += np.broadcast_to(grad, self.shape)

        out._backward = _backward
        return out

    def mean(self, axis: Optional[int] = None, keepdims: bool = False) -> "Tensor":
        if axis is None:
            n = self.data.size
        else:
            n = self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) / n

    def max(self, axis: Optional[int] = None, keepdims: bool = False) -> "Tensor":
        out = Tensor(
            self.data.max(axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="max",
        )

        def _backward():
            if self.requires_grad:
                max_vals = out.data
                if axis is not None and not keepdims:
                    max_vals = np.expand_dims(max_vals, axis=axis)
                mask = (self.data == np.broadcast_to(max_vals, self.shape)).astype(np.float32)
                grad = out.grad
                if axis is not None and not keepdims:
                    grad = np.expand_dims(grad, axis=axis)
                self.grad += mask * np.broadcast_to(grad, self.shape)

        out._backward = _backward
        return out

    # ─── Activation Functions ────────────────────────────────
    def relu(self) -> "Tensor":
        out = Tensor(
            np.maximum(0, self.data),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="relu",
        )

        def _backward():
            if self.requires_grad:
                self.grad += (self.data > 0).astype(np.float32) * out.grad

        out._backward = _backward
        return out

    def gelu(self) -> "Tensor":
        c = np.sqrt(2.0 / np.pi)
        inner = c * (self.data + 0.044715 * self.data ** 3)
        tanh_val = np.tanh(inner)
        out_data = 0.5 * self.data * (1.0 + tanh_val)
        out = Tensor(
            out_data,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="gelu",
        )

        def _backward():
            if self.requires_grad:
                sech2 = 1.0 - tanh_val ** 2
                d_inner = c * (1.0 + 3.0 * 0.044715 * self.data ** 2)
                grad = 0.5 * (1.0 + tanh_val) + 0.5 * self.data * sech2 * d_inner
                self.grad += grad * out.grad

        out._backward = _backward
        return out

    def silu(self) -> "Tensor":
        """SiLU / Swish activation: x * sigmoid(x)"""
        sig = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(
            self.data * sig,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="silu",
        )

        def _backward():
            if self.requires_grad:
                grad = sig * (1.0 + self.data * (1.0 - sig))
                self.grad += grad * out.grad

        out._backward = _backward
        return out

    def sigmoid(self) -> "Tensor":
        sig = 1.0 / (1.0 + np.exp(-np.clip(self.data, -500, 500)))
        out = Tensor(
            sig,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="sigmoid",
        )

        def _backward():
            if self.requires_grad:
                self.grad += sig * (1.0 - sig) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> "Tensor":
        t = np.tanh(self.data)
        out = Tensor(
            t,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="tanh",
        )

        def _backward():
            if self.requires_grad:
                self.grad += (1.0 - t ** 2) * out.grad

        out._backward = _backward
        return out

    # ─── Softmax & Log-Softmax ───────────────────────────────
    def softmax(self, axis: int = -1) -> "Tensor":
        shifted = self.data - self.data.max(axis=axis, keepdims=True)
        exp_vals = np.exp(shifted)
        sm = exp_vals / exp_vals.sum(axis=axis, keepdims=True)
        out = Tensor(
            sm,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="softmax",
        )

        def _backward():
            if self.requires_grad:
                s = sm
                ds = out.grad
                self.grad += s * (ds - (ds * s).sum(axis=axis, keepdims=True))

        out._backward = _backward
        return out

    def log_softmax(self, axis: int = -1) -> "Tensor":
        shifted = self.data - self.data.max(axis=axis, keepdims=True)
        log_sum_exp = np.log(np.exp(shifted).sum(axis=axis, keepdims=True))
        lsm = shifted - log_sum_exp
        out = Tensor(
            lsm,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="log_softmax",
        )

        def _backward():
            if self.requires_grad:
                sm = np.exp(lsm)
                self.grad += out.grad - sm * out.grad.sum(axis=axis, keepdims=True)

        out._backward = _backward
        return out

    # ─── Shape Operations ────────────────────────────────────
    def reshape(self, *shape) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        original_shape = self.shape
        out = Tensor(
            self.data.reshape(shape),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="reshape",
        )

        def _backward():
            if self.requires_grad:
                self.grad += out.grad.reshape(original_shape)

        out._backward = _backward
        return out

    def transpose(self, *axes) -> "Tensor":
        if not axes:
            axes_tuple = None
        elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes_tuple = tuple(axes[0])
        else:
            axes_tuple = axes

        out = Tensor(
            self.data.transpose(axes_tuple) if axes_tuple else self.data.T,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="transpose",
        )

        def _backward():
            if self.requires_grad:
                if axes_tuple:
                    inv = [0] * len(axes_tuple)
                    for i, a in enumerate(axes_tuple):
                        inv[a] = i
                    self.grad += out.grad.transpose(inv)
                else:
                    self.grad += out.grad.T

        out._backward = _backward
        return out

    def unsqueeze(self, axis: int) -> "Tensor":
        return Tensor(
            np.expand_dims(self.data, axis=axis),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="unsqueeze",
        )

    def squeeze(self, axis: Optional[int] = None) -> "Tensor":
        return Tensor(
            np.squeeze(self.data, axis=axis),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="squeeze",
        )

    # ─── Indexing ────────────────────────────────────────────
    def __getitem__(self, idx) -> "Tensor":
        out = Tensor(
            self.data[idx],
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="getitem",
        )

        def _backward():
            if self.requires_grad:
                full_grad = np.zeros_like(self.data)
                full_grad[idx] = out.grad
                self.grad += full_grad

        out._backward = _backward
        return out

    # ─── Loss Functions ──────────────────────────────────────
    def cross_entropy(self, targets: np.ndarray) -> "Tensor":
        """Cross-entropy loss with integrated log-softmax for stability."""
        shifted = self.data - self.data.max(axis=-1, keepdims=True)
        log_sum_exp = np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
        log_probs = shifted - log_sum_exp

        batch_size = targets.shape[0]
        if targets.ndim == 1:
            loss = -log_probs[np.arange(batch_size), targets].mean()
        else:
            seq_len = targets.shape[1]
            loss = -log_probs[
                np.arange(batch_size)[:, None],
                np.arange(seq_len)[None, :],
                targets,
            ].mean()

        out = Tensor(
            np.array(loss),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="cross_entropy",
        )

        def _backward():
            if self.requires_grad:
                sm = np.exp(log_probs)
                grad = sm.copy()
                n = batch_size * (targets.shape[1] if targets.ndim == 2 else 1)
                if targets.ndim == 1:
                    grad[np.arange(batch_size), targets] -= 1
                else:
                    seq_len = targets.shape[1]
                    grad[
                        np.arange(batch_size)[:, None],
                        np.arange(seq_len)[None, :],
                        targets,
                    ] -= 1
                self.grad += (grad / n) * out.grad

        out._backward = _backward
        return out

    # ─── Backpropagation ─────────────────────────────────────
    def backward(self):
        """Compute gradients via reverse-mode automatic differentiation."""
        topo: List[Tensor] = []
        visited = set()

        def build_topo(v: Tensor):
            if id(v) not in visited:
                visited.add(id(v))
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()

    def zero_grad(self):
        """Reset gradients to zero."""
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)

    # ─── Utility ─────────────────────────────────────────────
    def detach(self) -> "Tensor":
        return Tensor(self.data.copy())

    def numpy(self) -> np.ndarray:
        return self.data.copy()

    def item(self) -> float:
        return float(self.data)

    def clone(self) -> "Tensor":
        t = Tensor(self.data.copy(), requires_grad=self.requires_grad)
        if self.grad is not None:
            t.grad = self.grad.copy()
        return t

    def __repr__(self) -> str:
        return f"Tensor({self.data}, grad_fn={self._op or 'None'})"

    def __len__(self) -> int:
        return len(self.data)


# ─── Helper Functions ────────────────────────────────────────

def _unbroadcast(grad: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Sum out dimensions that were added by broadcasting."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, (gs, s) in enumerate(zip(grad.shape, shape)):
        if s == 1 and gs > 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


def _swap_last_two(arr: np.ndarray) -> np.ndarray:
    """Swap the last two axes of an array (batched transpose)."""
    axes = list(range(arr.ndim))
    axes[-1], axes[-2] = axes[-2], axes[-1]
    return arr.transpose(axes)


# ─── Factory Functions ───────────────────────────────────────

def zeros(*shape, requires_grad: bool = False) -> Tensor:
    return Tensor(np.zeros(shape, dtype=np.float32), requires_grad=requires_grad)


def ones(*shape, requires_grad: bool = False) -> Tensor:
    return Tensor(np.ones(shape, dtype=np.float32), requires_grad=requires_grad)


def randn(*shape, requires_grad: bool = False) -> Tensor:
    return Tensor(
        np.random.randn(*shape).astype(np.float32), requires_grad=requires_grad
    )


def rand(*shape, requires_grad: bool = False) -> Tensor:
    return Tensor(
        np.random.rand(*shape).astype(np.float32), requires_grad=requires_grad
    )


def from_numpy(arr: np.ndarray, requires_grad: bool = False) -> Tensor:
    return Tensor(arr, requires_grad=requires_grad)


def cat(tensors: List[Tensor], axis: int = 0) -> Tensor:
    out = Tensor(
        np.concatenate([t.data for t in tensors], axis=axis),
        requires_grad=any(t.requires_grad for t in tensors),
        _children=tuple(tensors),
        _op="cat",
    )

    def _backward():
        splits = np.cumsum([t.shape[axis] for t in tensors[:-1]])
        grads = np.split(out.grad, splits, axis=axis)
        for t, g in zip(tensors, grads):
            if t.requires_grad:
                t.grad += g

    out._backward = _backward
    return out


def stack(tensors: List[Tensor], axis: int = 0) -> Tensor:
    out = Tensor(
        np.stack([t.data for t in tensors], axis=axis),
        requires_grad=any(t.requires_grad for t in tensors),
        _children=tuple(tensors),
        _op="stack",
    )

    def _backward():
        grads = [
            np.take(out.grad, i, axis=axis) for i in range(len(tensors))
        ]
        for t, g in zip(tensors, grads):
            if t.requires_grad:
                t.grad += g

    out._backward = _backward
    return out
