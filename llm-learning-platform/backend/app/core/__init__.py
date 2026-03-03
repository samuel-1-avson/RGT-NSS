"""Core package for the LLM Learning Platform.

This package contains the foundational components:
- Tensor: Automatic differentiation engine
- Module: Neural network building blocks
- Optimizer: Training optimization algorithms
- State Management: Persistent storage
- Authentication: Security and access control
- Tokenization: Text processing
- WebSocket: Real-time communication
- Metrics: Monitoring and observability
"""

from .tensor import Tensor, cross_entropy_loss, mse_loss
from .module import Module, Linear, Embedding
from .optimizer import AdamW, Adam, SGD, clip_gradients, LinearWarmupCosineDecay
from .exceptions import (
    BaseAppException,
    ModelNotFoundError,
    ModelCreationError,
    TrainingSessionNotFoundError,
    TrainingError,
    InferenceError,
    AuthenticationError,
    AuthorizationError,
    ValidationError,
)
from .state_manager import (
    StateManager,
    MemoryStateManager,
    RedisStateManager,
    StateManagerFactory,
    BackendType,
    get_state_manager,
    set_state_manager,
)
from .auth import (
    User,
    create_access_token,
    decode_token,
    generate_api_key,
    verify_api_key,
    get_current_user,
    require_permissions,
)
from .tokenizer import (
    BaseTokenizer,
    CharacterTokenizer,
    WordTokenizer,
    BPETokenizer,
    TiktokenTokenizer,
    TokenizerFactory,
    get_default_tokenizer,
)
from .websocket_manager import (
    WebSocketConnectionManager,
    ConnectionInfo,
    get_connection_manager,
)
from .metrics import (
    MetricsCollector,
    TrainingMetrics,
    InferenceMetrics,
    timed,
    get_metrics_collector,
)

__all__ = [
    # Tensor & Module
    'Tensor', 'cross_entropy_loss', 'mse_loss',
    'Module', 'Linear', 'Embedding',
    # Optimizer
    'AdamW', 'Adam', 'SGD', 'clip_gradients', 'LinearWarmupCosineDecay',
    # Exceptions
    'BaseAppException',
    'ModelNotFoundError',
    'ModelCreationError',
    'TrainingSessionNotFoundError',
    'TrainingError',
    'InferenceError',
    'AuthenticationError',
    'AuthorizationError',
    'ValidationError',
    # State Management
    'StateManager',
    'MemoryStateManager',
    'RedisStateManager',
    'StateManagerFactory',
    'BackendType',
    'get_state_manager',
    'set_state_manager',
    # Auth
    'User',
    'create_access_token',
    'decode_token',
    'generate_api_key',
    'verify_api_key',
    'get_current_user',
    'require_permissions',
    # Tokenizer
    'BaseTokenizer',
    'CharacterTokenizer',
    'WordTokenizer',
    'BPETokenizer',
    'TiktokenTokenizer',
    'TokenizerFactory',
    'get_default_tokenizer',
    # WebSocket
    'WebSocketConnectionManager',
    'ConnectionInfo',
    'get_connection_manager',
    # Metrics
    'MetricsCollector',
    'TrainingMetrics',
    'InferenceMetrics',
    'timed',
    'get_metrics_collector',
]
