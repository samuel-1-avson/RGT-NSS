"""
Atomic GPT Implementation
Based on Andrej Karpathy's minimal, dependency-free GPT.
Adapted for the Interactive LLM Learning Platform.

This is the most atomic way to train and run inference for a GPT in pure, 
dependency-free Python. Everything else is just efficiency.

Key differences from full implementation:
- Pure Python (no NumPy) for maximum transparency
- Scalar operations only (easier to visualize step-by-step)
- Minimal dependencies (only random, math, os)
"""

import math
import random
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass


# =============================================================================
# AUTOMATIC DIFFERENTIATION ENGINE
# =============================================================================

class Value:
    """
    Scalar value with automatic differentiation support.
    
    This is the fundamental building block of our neural network.
    Each Value represents a node in the computation graph.
    """
    __slots__ = ('data', 'grad', '_children', '_local_grads', '_op', '_label')
    
    def __init__(self, data: float, children: Tuple['Value', ...] = (), 
                 local_grads: Tuple[float, ...] = (), op: str = '', label: str = ''):
        self.data = float(data)         # Forward pass value
        self.grad = 0.0                 # Gradient (computed in backward pass)
        self._children = children       # Parent nodes in computation graph
        self._local_grads = local_grads # Local derivatives ∂output/∂input
        self._op = op                   # Operation that created this node
        self._label = label             # For visualization/debugging
    
    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
    
    # -------------------- Arithmetic Operations --------------------
    
    def __add__(self, other: 'Value' | float) -> 'Value':
        other = other if isinstance(other, Value) else Value(other)
        # f(x, y) = x + y
        # ∂f/∂x = 1, ∂f/∂y = 1
        return Value(
            self.data + other.data,
            children=(self, other),
            local_grads=(1.0, 1.0),
            op='+'
        )
    
    def __mul__(self, other: 'Value' | float) -> 'Value':
        other = other if isinstance(other, Value) else Value(other)
        # f(x, y) = x * y
        # ∂f/∂x = y, ∂f/∂y = x
        return Value(
            self.data * other.data,
            children=(self, other),
            local_grads=(other.data, self.data),
            op='*'
        )
    
    def __pow__(self, other: float) -> 'Value':
        # f(x) = x^n
        # ∂f/∂x = n * x^(n-1)
        return Value(
            self.data ** other,
            children=(self,),
            local_grads=(other * self.data ** (other - 1),),
            op=f'**{other}'
        )
    
    def __neg__(self) -> 'Value':
        return self * -1
    
    def __radd__(self, other: float) -> 'Value':
        return self + other
    
    def __sub__(self, other: 'Value' | float) -> 'Value':
        return self + (-other)
    
    def __rsub__(self, other: float) -> 'Value':
        return other + (-self)
    
    def __rmul__(self, other: float) -> 'Value':
        return self * other
    
    def __truediv__(self, other: 'Value' | float) -> 'Value':
        return self * (other ** -1 if isinstance(other, Value) else Value(other) ** -1)
    
    # -------------------- Activation Functions --------------------
    
    def relu(self) -> 'Value':
        """ReLU activation: f(x) = max(0, x)"""
        # ∂f/∂x = 1 if x > 0 else 0
        return Value(
            max(0.0, self.data),
            children=(self,),
            local_grads=(float(self.data > 0),),
            op='relu'
        )
    
    def gelu(self) -> 'Value':
        """GELU activation approximation"""
        # GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x^3)))
        # Simplified for this atomic implementation
        return Value(
            self.data * 0.5 * (1.0 + math.tanh(
                math.sqrt(2.0 / math.pi) * (self.data + 0.044715 * self.data ** 3)
            )),
            children=(self,),
            local_grads=(1.0,),  # Approximate gradient
            op='gelu'
        )
    
    def exp(self) -> 'Value':
        """Exponential: f(x) = e^x"""
        # ∂f/∂x = e^x = f(x)
        out = math.exp(self.data)
        return Value(
            out,
            children=(self,),
            local_grads=(out,),
            op='exp'
        )
    
    def log(self) -> 'Value':
        """Natural logarithm: f(x) = ln(x)"""
        # ∂f/∂x = 1/x
        return Value(
            math.log(self.data + 1e-8),  # Small epsilon for stability
            children=(self,),
            local_grads=(1.0 / (self.data + 1e-8),),
            op='log'
        )
    
    # -------------------- Backpropagation --------------------
    
    def backward(self):
        """
        Reverse-mode automatic differentiation (backpropagation).
        
        1. Build topological order of computation graph
        2. Initialize output gradient to 1
        3. Apply chain rule in reverse order
        """
        # Topological sort
        topo: List[Value] = []
        visited: set = set()
        
        def build_topo(v: Value):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        
        # Seed gradient
        self.grad = 1.0
        
        # Backpropagate in reverse topological order
        for node in reversed(topo):
            for child, local_grad in zip(node._children, node._local_grads):
                # Chain rule: ∂loss/∂child = ∂loss/∂node * ∂node/∂child
                child.grad += local_grad * node.grad


# =============================================================================
# NEURAL NETWORK LAYERS
# =============================================================================

def matrix(rows: int, cols: int, std: float = 0.08) -> List[List[Value]]:
    """Create a matrix of Values with Gaussian initialization."""
    return [[Value(random.gauss(0.0, std)) for _ in range(cols)] for _ in range(rows)]


def linear(x: List[Value], w: List[List[Value]]) -> List[Value]:
    """
    Linear transformation: y = x @ W^T
    
    Args:
        x: Input vector of shape (in_features,)
        w: Weight matrix of shape (out_features, in_features)
    
    Returns:
        Output vector of shape (out_features,)
    """
    return [sum(wi * xi for wi, xi in zip(row, x)) for row in w]


def softmax(logits: List[Value]) -> List[Value]:
    """
    Softmax normalization for probability distribution.
    
    Subtract max for numerical stability.
    """
    # Numerical stability: subtract max
    max_val = max(val.data for val in logits)
    shifted = [val - max_val for val in logits]
    
    # Exponentiate
    exps = [val.exp() for val in shifted]
    
    # Normalize
    total = sum(exps)
    return [e / total for e in exps]


def rmsnorm(x: List[Value]) -> List[Value]:
    """
    Root Mean Square Layer Normalization.
    
    Modern alternative to LayerNorm (used in Llama, etc.):
    RMSNorm(x) = x / sqrt(mean(x^2) + eps)
    """
    # Calculate mean of squares
    ms = sum(xi * xi for xi in x) / len(x)
    
    # Scale factor
    scale = (ms + 1e-5) ** -0.5
    
    # Normalize
    return [xi * scale for xi in x]


# =============================================================================
# GPT MODEL
# =============================================================================

@dataclass
class AtomicGPTConfig:
    """Configuration for Atomic GPT model."""
    vocab_size: int = 27      # Number of unique tokens (+1 for BOS)
    n_layer: int = 1          # Number of transformer layers
    n_embd: int = 16          # Embedding dimension
    block_size: int = 16      # Maximum sequence length (context window)
    n_head: int = 4           # Number of attention heads
    dropout: float = 0.0      # Dropout rate (not implemented in atomic version)


class AtomicGPT:
    """
    Minimal GPT model implementation.
    
    Architecture follows GPT-2 with minor differences:
    - LayerNorm -> RMSNorm
    - No biases
    - GeLU -> ReLU (configurable)
    """
    
    def __init__(self, config: AtomicGPTConfig):
        self.config = config
        self.head_dim = config.n_embd // config.n_head
        
        # Initialize state dictionary
        self.state_dict: Dict[str, List[List[Value]]] = {}
        
        # Token and position embeddings
        std = 0.08
        self.state_dict['wte'] = matrix(config.vocab_size, config.n_embd, std)
        self.state_dict['wpe'] = matrix(config.block_size, config.n_embd, std)
        self.state_dict['lm_head'] = matrix(config.vocab_size, config.n_embd, std)
        
        # Transformer layers
        for i in range(config.n_layer):
            # Attention weights
            self.state_dict[f'layer{i}.attn_wq'] = matrix(config.n_embd, config.n_embd, std)
            self.state_dict[f'layer{i}.attn_wk'] = matrix(config.n_embd, config.n_embd, std)
            self.state_dict[f'layer{i}.attn_wv'] = matrix(config.n_embd, config.n_embd, std)
            self.state_dict[f'layer{i}.attn_wo'] = matrix(config.n_embd, config.n_embd, std)
            
            # MLP weights
            self.state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * config.n_embd, config.n_embd, std)
            self.state_dict[f'layer{i}.mlp_fc2'] = matrix(config.n_embd, 4 * config.n_embd, std)
        
        # Flatten all parameters for optimizer
        self.params: List[Value] = [
            p for mat in self.state_dict.values() 
            for row in mat 
            for p in row
        ]
    
    def forward(self, token_id: int, pos_id: int, 
                keys: List[List[List[Value]]], 
                values: List[List[List[Value]]]) -> List[Value]:
        """
        Forward pass for a single token position.
        
        Args:
            token_id: ID of current token
            pos_id: Position in sequence
            keys: KV cache for keys (per layer, per position)
            values: KV cache for values (per layer, per position)
        
        Returns:
            Logits over vocabulary
        """
        config = self.config
        
        # 1. Embeddings
        tok_emb = self.state_dict['wte'][token_id]
        pos_emb = self.state_dict['wpe'][pos_id]
        x = [t + p for t, p in zip(tok_emb, pos_emb)]
        
        # 2. Transformer blocks
        for li in range(config.n_layer):
            # ---- Multi-Head Attention ----
            x_residual = x
            x = rmsnorm(x)
            
            # Project to Q, K, V
            q = linear(x, self.state_dict[f'layer{li}.attn_wq'])
            k = linear(x, self.state_dict[f'layer{li}.attn_wk'])
            v = linear(x, self.state_dict[f'layer{li}.attn_wv'])
            
            # Store K, V for future positions
            keys[li].append(k)
            values[li].append(v)
            
            # Multi-head attention computation
            x_attn = []
            for h in range(config.n_head):
                hs = h * self.head_dim
                
                # Split into heads
                q_h = q[hs:hs + self.head_dim]
                k_h = [ki[hs:hs + self.head_dim] for ki in keys[li]]
                v_h = [vi[hs:hs + self.head_dim] for vi in values[li]]
                
                # Attention scores: Q @ K^T / sqrt(d_k)
                attn_logits = [
                    sum(q_h[j] * k_h[t][j] for j in range(self.head_dim)) / (self.head_dim ** 0.5)
                    for t in range(len(k_h))
                ]
                
                # Softmax to get attention weights
                attn_weights = softmax(attn_logits)
                
                # Weighted sum of values
                head_out = [
                    sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                    for j in range(self.head_dim)
                ]
                
                x_attn.extend(head_out)
            
            # Output projection
            x = linear(x_attn, self.state_dict[f'layer{li}.attn_wo'])
            
            # Residual connection
            x = [a + b for a, b in zip(x, x_residual)]
            
            # ---- Feed-Forward Network (MLP) ----
            x_residual = x
            x = rmsnorm(x)
            
            # Expand
            x = linear(x, self.state_dict[f'layer{li}.mlp_fc1'])
            
            # Activation
            x = [xi.relu() for xi in x]
            
            # Project back
            x = linear(x, self.state_dict[f'layer{li}.mlp_fc2'])
            
            # Residual connection
            x = [a + b for a, b in zip(x, x_residual)]
        
        # 3. Output projection to vocabulary
        logits = linear(x, self.state_dict['lm_head'])
        
        return logits
    
    def count_parameters(self) -> int:
        """Return total number of parameters."""
        return len(self.params)


# =============================================================================
# OPTIMIZER
# =============================================================================

class Adam:
    """
    Adam optimizer with linear learning rate decay.
    
    Adam maintains per-parameter adaptive learning rates based on 
    first and second moment estimates of gradients.
    """
    
    def __init__(self, params: List[Value], lr: float = 0.01, 
                 beta1: float = 0.85, beta2: float = 0.99, eps: float = 1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        
        # Initialize moment buffers
        self.m = [0.0] * len(params)  # First moment (mean)
        self.v = [0.0] * len(params)  # Second moment (uncentered variance)
    
    def step(self, current_step: int, total_steps: int):
        """Single optimization step."""
        self.t += 1
        
        # Linear learning rate decay
        lr_t = self.lr * (1 - current_step / total_steps)
        
        for i, p in enumerate(self.params):
            if p.grad == 0:
                continue
            
            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
            
            # Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (p.grad ** 2)
            
            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Update parameter
            p.data -= lr_t * m_hat / (v_hat ** 0.5 + self.eps)
            
            # Reset gradient for next iteration
            p.grad = 0.0


# =============================================================================
# TRAINING
# =============================================================================

class Trainer:
    """Training loop for Atomic GPT."""
    
    def __init__(self, model: AtomicGPT, optimizer: Adam):
        self.model = model
        self.optimizer = optimizer
        self.step = 0
        self.loss_history: List[float] = []
    
    def train_step(self, tokens: List[int]) -> float:
        """
        Single training step on a sequence of tokens.
        
        Args:
            tokens: List of token IDs (including BOS markers)
        
        Returns:
            Loss value
        """
        config = self.model.config
        n = min(config.block_size, len(tokens) - 1)
        
        # KV caches for attention
        keys: List[List[List[Value]]] = [[] for _ in range(config.n_layer)]
        values: List[List[List[Value]]] = [[] for _ in range(config.n_layer)]
        
        losses = []
        
        # Forward pass for each position
        for pos_id in range(n):
            token_id = tokens[pos_id]
            target_id = tokens[pos_id + 1]
            
            # Forward pass
            logits = self.model.forward(token_id, pos_id, keys, values)
            
            # Softmax to get probabilities
            probs = softmax(logits)
            
            # Cross-entropy loss: -log(p[target])
            loss_t = -probs[target_id].log()
            losses.append(loss_t)
        
        # Average loss over sequence
        loss = (1.0 / n) * sum(losses)
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        self.optimizer.step(self.step, 1000)  # Assuming 1000 total steps
        
        self.step += 1
        self.loss_history.append(loss.data)
        
        return loss.data
    
    def generate(self, max_length: int = 16, temperature: float = 0.5, 
                 seed_token: int = None) -> List[int]:
        """
        Generate a sequence of tokens.
        
        Args:
            max_length: Maximum sequence length
            temperature: Sampling temperature (0-1, lower = more deterministic)
            seed_token: Starting token ID (default: BOS)
        
        Returns:
            List of generated token IDs
        """
        config = self.model.config
        BOS = config.vocab_size - 1
        
        keys = [[] for _ in range(config.n_layer)]
        values = [[] for _ in range(config.n_layer)]
        
        token_id = seed_token if seed_token is not None else BOS
        generated = []
        
        for pos_id in range(max_length):
            # Forward pass
            logits = self.model.forward(token_id, pos_id, keys, values)
            
            # Apply temperature
            scaled_logits = [l / temperature for l in logits]
            
            # Softmax to get probabilities
            probs = softmax(scaled_logits)
            
            # Sample next token
            probs_data = [p.data for p in probs]
            token_id = random.choices(range(config.vocab_size), weights=probs_data)[0]
            
            # Stop if BOS token
            if token_id == BOS:
                break
            
            generated.append(token_id)
        
        return generated


# =============================================================================
# DATASET
# =============================================================================

class CharacterDataset:
    """Simple character-level dataset."""
    
    def __init__(self, texts: List[str]):
        self.texts = texts
        
        # Build vocabulary
        chars = sorted(set(''.join(texts)))
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for ch, i in self.char_to_idx.items()}
        
        # BOS token
        self.BOS = len(chars)
        self.vocab_size = len(chars) + 1
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        return [self.char_to_idx[ch] for ch in text if ch in self.char_to_idx]
    
    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to text."""
        return ''.join(self.idx_to_char.get(t, '?') for t in tokens)
    
    def get_sample(self) -> List[int]:
        """Get a random training sample."""
        text = random.choice(self.texts)
        tokens = [self.BOS] + self.encode(text) + [self.BOS]
        return tokens


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def demo():
    """Demonstrate the atomic GPT."""
    # Sample data (names)
    names = [
        "emma", "olivia", "ava", "isabella", "sophia",
        "liam", "noah", "oliver", "elijah", "james",
        "alex", "sam", "jordan", "taylor", "morgan"
    ]
    
    # Create dataset
    dataset = CharacterDataset(names)
    print(f"Vocabulary: {dataset.idx_to_char}")
    print(f"Vocab size: {dataset.vocab_size}")
    
    # Create model
    config = AtomicGPTConfig(
        vocab_size=dataset.vocab_size,
        n_layer=1,
        n_embd=16,
        block_size=16,
        n_head=4
    )
    model = AtomicGPT(config)
    print(f"Parameters: {model.count_parameters()}")
    
    # Create optimizer
    optimizer = Adam(model.params, lr=0.01)
    
    # Create trainer
    trainer = Trainer(model, optimizer)
    
    # Train
    print("\nTraining...")
    for step in range(100):
        tokens = dataset.get_sample()
        loss = trainer.train_step(tokens)
        if step % 20 == 0:
            print(f"Step {step:3d} | Loss: {loss:.4f}")
    
    # Generate
    print("\nGenerating names:")
    for i in range(10):
        tokens = trainer.generate(max_length=10, temperature=0.5)
        name = dataset.decode(tokens)
        print(f"  {i+1:2d}. {name}")


if __name__ == "__main__":
    demo()
