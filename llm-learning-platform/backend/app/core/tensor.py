"""
Custom Automatic Differentiation Engine
Built from scratch for educational transparency
"""

import numpy as np
from typing import List, Tuple, Optional, Callable, Set, Union
import json


class Tensor:
    """
    Tensor with automatic differentiation support.
    
    This is the core data structure for our educational deep learning framework.
    It wraps a numpy array and tracks operations for backpropagation.
    """
    
    def __init__(
        self, 
        data: np.ndarray, 
        children: Tuple['Tensor', ...] = (),
        op: str = '',
        label: str = '',
        requires_grad: bool = True
    ):
        self.data = np.array(data, dtype=np.float32)
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self._backward = lambda: None
        self._prev = set(children)
        self._op = op
        self._label = label
        self.shape = self.data.shape
        self.requires_grad = requires_grad
        
    def __repr__(self):
        return f"Tensor(shape={self.shape}, grad_fn={self._op}, requires_grad={self.requires_grad})"
    
    def __add__(self, other: Union['Tensor', float, int, np.ndarray]) -> 'Tensor':
        other = other if isinstance(other, Tensor) else Tensor(np.array(other), requires_grad=False)
        out = Tensor(
            self.data + other.data, 
            (self, other), 
            '+',
            requires_grad=self.requires_grad or other.requires_grad
        )
        
        def _backward():
            if self.requires_grad:
                # Sum grad over broadcasted dimensions if needed
                grad = out.grad
                if self.shape != out.grad.shape:
                    grad = self._sum_to_shape(out.grad, self.shape)
                self.grad += grad
            if other.requires_grad:
                grad = out.grad
                if other.shape != out.grad.shape:
                    grad = self._sum_to_shape(out.grad, other.shape)
                other.grad += grad
        out._backward = _backward
        
        return out
    
    def __radd__(self, other: Union[float, int, np.ndarray]) -> 'Tensor':
        return self.__add__(other)
    
    def __sub__(self, other: Union['Tensor', float, int, np.ndarray]) -> 'Tensor':
        return self.__add__(-other)
    
    def __rsub__(self, other: Union[float, int, np.ndarray]) -> 'Tensor':
        return (-self).__add__(other)
    
    def __neg__(self) -> 'Tensor':
        return self.__mul__(-1)
    
    def __mul__(self, other: Union['Tensor', float, int, np.ndarray]) -> 'Tensor':
        other = other if isinstance(other, Tensor) else Tensor(np.array(other), requires_grad=False)
        out = Tensor(
            self.data * other.data, 
            (self, other), 
            '*',
            requires_grad=self.requires_grad or other.requires_grad
        )
        
        def _backward():
            if self.requires_grad:
                grad = other.data * out.grad
                if self.shape != grad.shape:
                    grad = self._sum_to_shape(grad, self.shape)
                self.grad += grad
            if other.requires_grad:
                grad = self.data * out.grad
                if other.shape != grad.shape:
                    grad = self._sum_to_shape(grad, other.shape)
                other.grad += grad
        out._backward = _backward
        
        return out
    
    def __rmul__(self, other: Union[float, int, np.ndarray]) -> 'Tensor':
        return self.__mul__(other)
    
    def __truediv__(self, other: Union['Tensor', float, int, np.ndarray]) -> 'Tensor':
        return self.__mul__(other ** -1)
    
    def __pow__(self, other: Union[float, int]) -> 'Tensor':
        assert isinstance(other, (int, float)), "Only scalar powers supported"
        out = Tensor(
            self.data ** other, 
            (self,), 
            f'**{other}',
            requires_grad=self.requires_grad
        )
        
        def _backward():
            if self.requires_grad:
                self.grad += (other * self.data ** (other - 1)) * out.grad
        out._backward = _backward
        
        return out
    
    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(
            self.data @ other.data, 
            (self, other), 
            '@',
            requires_grad=self.requires_grad or other.requires_grad
        )
        
        def _backward():
            if self.requires_grad:
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other.grad += self.data.T @ out.grad
        out._backward = _backward
        
        return out
    
    def sum(self, dim: Optional[int] = None, keepdim: bool = False) -> 'Tensor':
        """Sum elements along dimension."""
        out_data = self.data.sum(axis=dim, keepdims=keepdim)
        out = Tensor(out_data, (self,), 'sum', requires_grad=self.requires_grad)
        
        def _backward():
            if self.requires_grad:
                if dim is None:
                    self.grad += np.ones_like(self.data) * out.grad
                else:
                    shape = list(self.shape)
                    if not keepdim:
                        shape[dim] = 1
                    grad = out.grad.reshape(shape)
                    self.grad += np.broadcast_to(grad, self.shape)
        out._backward = _backward
        
        return out
    
    def mean(self, dim: Optional[int] = None, keepdim: bool = False) -> 'Tensor':
        """Mean of elements along dimension."""
        if dim is None:
            n = self.data.size
        else:
            n = self.shape[dim]
        return self.sum(dim=dim, keepdim=keepdim) / n
    
    def reshape(self, *shape: int) -> 'Tensor':
        """Reshape tensor."""
        out = Tensor(
            self.data.reshape(shape), 
            (self,), 
            'reshape',
            requires_grad=self.requires_grad
        )
        
        def _backward():
            if self.requires_grad:
                self.grad += out.grad.reshape(self.shape)
        out._backward = _backward
        
        return out
    
    def transpose(self, *axes: int) -> 'Tensor':
        """Transpose tensor dimensions."""
        out = Tensor(
            self.data.transpose(axes) if axes else self.data.T, 
            (self,), 
            'transpose',
            requires_grad=self.requires_grad
        )
        
        def _backward():
            if self.requires_grad:
                self.grad += out.grad.transpose(axes) if axes else out.grad.T
        out._backward = _backward
        
        return out
    
    def softmax(self, dim: int = -1) -> 'Tensor':
        """Softmax with numerical stability."""
        # Subtract max for numerical stability
        max_val = np.max(self.data, axis=dim, keepdims=True)
        exp_x = np.exp(self.data - max_val)
        probs = exp_x / np.sum(exp_x, axis=dim, keepdims=True)
        
        out = Tensor(probs, (self,), 'softmax', requires_grad=self.requires_grad)
        
        def _backward():
            if self.requires_grad:
                # Softmax Jacobian: diag(p) - p @ p.T
                # Simplified for common cases
                self.grad += probs * (out.grad - np.sum(out.grad * probs, axis=dim, keepdims=True))
        out._backward = _backward
        
        return out
    
    def log(self) -> 'Tensor':
        """Natural logarithm."""
        out = Tensor(
            np.log(self.data + 1e-8), 
            (self,), 
            'log',
            requires_grad=self.requires_grad
        )
        
        def _backward():
            if self.requires_grad:
                self.grad += out.grad / (self.data + 1e-8)
        out._backward = _backward
        
        return out
    
    def exp(self) -> 'Tensor':
        """Exponential."""
        out_data = np.exp(self.data)
        out = Tensor(out_data, (self,), 'exp', requires_grad=self.requires_grad)
        
        def _backward():
            if self.requires_grad:
                self.grad += out_data * out.grad
        out._backward = _backward
        
        return out
    
    def sqrt(self) -> 'Tensor':
        """Square root."""
        return self.__pow__(0.5)
    
    def relu(self) -> 'Tensor':
        """ReLU activation."""
        out = Tensor(
            np.maximum(0, self.data), 
            (self,), 
            'relu',
            requires_grad=self.requires_grad
        )
        
        def _backward():
            if self.requires_grad:
                self.grad += (self.data > 0).astype(np.float32) * out.grad
        out._backward = _backward
        
        return out
    
    def gelu(self) -> 'Tensor':
        """GELU activation approximation."""
        # Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        sqrt_2_over_pi = np.sqrt(2 / np.pi)
        x_cubed = self.data ** 3
        tanh_arg = sqrt_2_over_pi * (self.data + 0.044715 * x_cubed)
        tanh_val = np.tanh(tanh_arg)
        
        out_data = 0.5 * self.data * (1 + tanh_val)
        out = Tensor(out_data, (self,), 'gelu', requires_grad=self.requires_grad)
        
        def _backward():
            if self.requires_grad:
                # Derivative of GELU
                sech_sq = 1 / np.cosh(tanh_arg) ** 2
                local_grad = (0.5 * (1 + tanh_val) + 
                             0.5 * self.data * sech_sq * sqrt_2_over_pi * 
                             (1 + 3 * 0.044715 * self.data ** 2))
                self.grad += local_grad * out.grad
        out._backward = _backward
        
        return out
    
    def tanh(self) -> 'Tensor':
        """Hyperbolic tangent."""
        out_data = np.tanh(self.data)
        out = Tensor(out_data, (self,), 'tanh', requires_grad=self.requires_grad)
        
        def _backward():
            if self.requires_grad:
                self.grad += (1 - out_data ** 2) * out.grad
        out._backward = _backward
        
        return out
    
    def dropout(self, p: float = 0.5, training: bool = True) -> 'Tensor':
        """Dropout regularization."""
        if not training or p == 0:
            return self
        
        mask = (np.random.rand(*self.shape) > p).astype(np.float32) / (1 - p)
        out = Tensor(
            self.data * mask, 
            (self,), 
            'dropout',
            requires_grad=self.requires_grad
        )
        
        def _backward():
            if self.requires_grad:
                self.grad += mask * out.grad
        out._backward = _backward
        
        return out
    
    def backward(self):
        """
        Reverse-mode automatic differentiation (backpropagation).
        Computes gradients by traversing the computation graph.
        """
        if not self.requires_grad:
            return
        
        # Topological sort
        topo = []
        visited = set()
        
        def build_topo(v: 'Tensor'):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        
        # Seed gradient
        self.grad = np.ones_like(self.data)
        
        # Backpropagate in reverse topological order
        for node in reversed(topo):
            node._backward()
    
    def zero_grad(self):
        """Zero out gradients."""
        if self.grad is not None:
            self.grad = np.zeros_like(self.data)
    
    def detach(self) -> 'Tensor':
        """Return a new tensor detached from the computation graph."""
        return Tensor(self.data, requires_grad=False)
    
    def numpy(self) -> np.ndarray:
        """Return numpy array (detached)."""
        return self.data.copy()
    
    def item(self) -> float:
        """Return scalar value."""
        return float(self.data)
    
    def _sum_to_shape(self, grad: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
        """Sum gradient to match target shape (for broadcasting)."""
        while len(grad.shape) > len(target_shape):
            grad = grad.sum(axis=0)
        
        for i, (grad_dim, target_dim) in enumerate(zip(grad.shape, target_shape)):
            if target_dim == 1 and grad_dim != 1:
                grad = grad.sum(axis=i, keepdims=True)
        
        return grad
    
    @classmethod
    def zeros(cls, *shape: int, requires_grad: bool = True) -> 'Tensor':
        """Create tensor filled with zeros."""
        return cls(np.zeros(shape), requires_grad=requires_grad)
    
    @classmethod
    def ones(cls, *shape: int, requires_grad: bool = True) -> 'Tensor':
        """Create tensor filled with ones."""
        return cls(np.ones(shape), requires_grad=requires_grad)
    
    @classmethod
    def randn(cls, *shape: int, requires_grad: bool = True) -> 'Tensor':
        """Create tensor with random normal values."""
        return cls(np.random.randn(*shape), requires_grad=requires_grad)
    
    @classmethod
    def rand(cls, *shape: int, requires_grad: bool = True) -> 'Tensor':
        """Create tensor with random uniform values."""
        return cls(np.random.rand(*shape), requires_grad=requires_grad)
    
    @classmethod
    def xavier_uniform(cls, *shape: int, requires_grad: bool = True) -> 'Tensor':
        """Xavier/Glorot uniform initialization."""
        if len(shape) < 2:
            limit = np.sqrt(6.0)
        else:
            fan_in, fan_out = shape[-2], shape[-1]
            limit = np.sqrt(6.0 / (fan_in + fan_out))
        return cls(np.random.uniform(-limit, limit, shape), requires_grad=requires_grad)
    
    @classmethod
    def kaiming_normal(cls, *shape: int, requires_grad: bool = True) -> 'Tensor':
        """Kaiming/He normal initialization."""
        if len(shape) < 2:
            std = np.sqrt(2.0)
        else:
            fan_in = np.prod(shape[1:])
            std = np.sqrt(2.0 / fan_in)
        return cls(np.random.randn(*shape) * std, requires_grad=requires_grad)


def cross_entropy_loss(logits: Tensor, targets: np.ndarray) -> Tensor:
    """
    Cross-entropy loss for classification.
    
    Args:
        logits: Raw model outputs (batch_size, num_classes)
        targets: Integer class labels (batch_size,)
    
    Returns:
        Scalar loss tensor
    """
    batch_size = logits.shape[0]
    
    # Log-softmax for numerical stability
    max_logits = np.max(logits.data, axis=-1, keepdims=True)
    exp_logits = np.exp(logits.data - max_logits)
    log_probs = logits.data - max_logits - np.log(np.sum(exp_logits, axis=-1, keepdims=True))
    
    # Negative log likelihood
    nll = -log_probs[np.arange(batch_size), targets]
    loss = np.mean(nll)
    
    out = Tensor(loss, (logits,), 'cross_entropy', requires_grad=logits.requires_grad)
    
    def _backward():
        if logits.requires_grad:
            # Gradient of cross-entropy with softmax
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
            probs[np.arange(batch_size), targets] -= 1
            logits.grad += (probs / batch_size) * out.grad
    out._backward = _backward
    
    return out


def mse_loss(predictions: Tensor, targets: Tensor) -> Tensor:
    """Mean squared error loss."""
    diff = predictions - targets
    loss = (diff * diff).mean()
    return loss


def cosine_similarity(a: Tensor, b: Tensor, dim: int = -1, eps: float = 1e-8) -> Tensor:
    """Cosine similarity between tensors."""
    a_norm = np.sqrt((a.data ** 2).sum(axis=dim, keepdims=True))
    b_norm = np.sqrt((b.data ** 2).sum(axis=dim, keepdims=True))
    
    similarity = (a.data * b.data).sum(axis=dim, keepdims=True) / (a_norm * b_norm + eps)
    
    return Tensor(similarity, requires_grad=False)
