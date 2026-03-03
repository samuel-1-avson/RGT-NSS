"""
Base module class for neural network components.
"""

from typing import List, Dict, Any, Optional
from .tensor import Tensor
import numpy as np


class Module:
    """
    Base class for all neural network modules.
    Similar to PyTorch's nn.Module.
    """
    
    def __init__(self):
        self._parameters: Dict[str, Tensor] = {}
        self._modules: Dict[str, 'Module'] = {}
        self._buffers: Dict[str, np.ndarray] = {}
        self.training = True
        
    def __call__(self, *args, **kwargs):
        """Forward pass when called."""
        return self.forward(*args, **kwargs)
    
    def forward(self, *args, **kwargs):
        """Forward pass - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement forward()")
    
    def __setattr__(self, name: str, value):
        """Track parameters and submodules."""
        if isinstance(value, Tensor):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        super().__setattr__(name, value)
    
    def parameters(self) -> List[Tensor]:
        """Get all trainable parameters."""
        params = list(self._parameters.values())
        for module in self._modules.values():
            params.extend(module.parameters())
        return params
    
    def named_parameters(self) -> Dict[str, Tensor]:
        """Get named parameters."""
        params = {}
        for name, param in self._parameters.items():
            params[name] = param
        for module_name, module in self._modules.items():
            for name, param in module.named_parameters().items():
                params[f"{module_name}.{name}"] = param
        return params
    
    def modules(self) -> List['Module']:
        """Get all submodules."""
        mods = []
        for module in self._modules.values():
            mods.append(module)
            mods.extend(module.modules())
        return mods
    
    def zero_grad(self):
        """Zero all parameter gradients."""
        for p in self.parameters():
            p.zero_grad()
    
    def train(self, mode: bool = True):
        """Set training mode."""
        self.training = mode
        for module in self._modules.values():
            module.train(mode)
        return self
    
    def eval(self):
        """Set evaluation mode."""
        return self.train(False)
    
    def state_dict(self) -> Dict[str, Any]:
        """Get state dictionary for saving."""
        state = {}
        for name, param in self._parameters.items():
            state[name] = param.data.copy()
        for name, buffer in self._buffers.items():
            state[name] = buffer.copy()
        for module_name, module in self._modules.items():
            state[module_name] = module.state_dict()
        return state
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state from dictionary."""
        for name, param in self._parameters.items():
            if name in state_dict:
                param.data = state_dict[name].copy()
        for name, buffer in self._buffers.items():
            if name in state_dict:
                self._buffers[name] = state_dict[name].copy()
        for module_name, module in self._modules.items():
            if module_name in state_dict:
                module.load_state_dict(state_dict[module_name])
    
    def count_parameters(self) -> int:
        """Count total number of parameters."""
        return sum(p.data.size for p in self.parameters())
    
    def count_trainable_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.data.size for p in self.parameters() if p.requires_grad)


class Linear(Module):
    """Linear transformation layer: y = xA^T + b"""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Xavier initialization
        limit = np.sqrt(6.0 / (in_features + out_features))
        self.weight = Tensor.xavier_uniform(in_features, out_features)
        
        if bias:
            self.bias = Tensor.zeros(out_features)
        else:
            self.bias = None
    
    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


class Embedding(Module):
    """Embedding layer."""
    
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # Initialize embeddings
        self.weight = Tensor.randn(num_embeddings, embedding_dim) * 0.02
    
    def forward(self, indices: np.ndarray) -> Tensor:
        """
        Args:
            indices: (batch_size, seq_len) integer array
        Returns:
            embeddings: (batch_size, seq_len, embedding_dim)
        """
        return Tensor(self.weight.data[indices], requires_grad=True)


class Dropout(Module):
    """Dropout layer for regularization."""
    
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p
    
    def forward(self, x: Tensor) -> Tensor:
        return x.dropout(self.p, self.training)
