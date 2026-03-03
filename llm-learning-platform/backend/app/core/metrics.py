"""
Monitoring and metrics for the LLM Learning Platform.
Provides Prometheus-compatible metrics and custom monitoring.
"""

import time
import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from contextlib import contextmanager
import functools
import asyncio

try:
    from prometheus_client import Counter, Histogram, Gauge, Info, CollectorRegistry
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


logger = logging.getLogger(__name__)


# =============================================================================
# Prometheus Metrics (if available)
# =============================================================================

if PROMETHEUS_AVAILABLE:
    # Info
    APP_INFO = Info('llm_platform_info', 'Application information')
    
    # Counters
    MODEL_CREATIONS = Counter(
        'llm_models_created_total',
        'Total number of models created',
        ['backend', 'status']
    )
    MODEL_DELETIONS = Counter(
        'llm_models_deleted_total',
        'Total number of models deleted'
    )
    TRAINING_SESSIONS_STARTED = Counter(
        'llm_training_sessions_started_total',
        'Total training sessions started',
        ['backend']
    )
    TRAINING_SESSIONS_COMPLETED = Counter(
        'llm_training_sessions_completed_total',
        'Total training sessions completed',
        ['backend', 'status']
    )
    INFERENCE_REQUESTS = Counter(
        'llm_inference_requests_total',
        'Total inference requests',
        ['model_type', 'status']
    )
    API_REQUESTS = Counter(
        'llm_api_requests_total',
        'Total API requests',
        ['method', 'endpoint', 'status']
    )
    WEBSOCKET_CONNECTIONS = Counter(
        'llm_websocket_connections_total',
        'Total WebSocket connections',
        ['event_type']
    )
    
    # Histograms
    MODEL_CREATION_DURATION = Histogram(
        'llm_model_creation_duration_seconds',
        'Time spent creating models',
        ['backend']
    )
    TRAINING_STEP_DURATION = Histogram(
        'llm_training_step_duration_seconds',
        'Time per training step',
        ['backend']
    )
    INFERENCE_DURATION = Histogram(
        'llm_inference_duration_seconds',
        'Time spent on inference',
        buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    )
    API_REQUEST_DURATION = Histogram(
        'llm_api_request_duration_seconds',
        'API request duration',
        ['method', 'endpoint']
    )
    
    # Gauges
    ACTIVE_MODELS = Gauge(
        'llm_active_models',
        'Number of active models',
        ['backend']
    )
    ACTIVE_TRAINING_SESSIONS = Gauge(
        'llm_active_training_sessions',
        'Number of active training sessions',
        ['backend']
    )
    GPU_MEMORY_USED = Gauge(
        'llm_gpu_memory_used_bytes',
        'GPU memory used',
        ['device']
    )
    GPU_UTILIZATION = Gauge(
        'llm_gpu_utilization_percent',
        'GPU utilization percentage',
        ['device']
    )
    WEBSOCKET_ACTIVE_CONNECTIONS = Gauge(
        'llm_websocket_active_connections',
        'Number of active WebSocket connections',
        ['session_type']
    )


# =============================================================================
# Custom Metrics
# =============================================================================

@dataclass
class TrainingMetrics:
    """Training session metrics."""
    session_id: str
    model_id: str
    step: int
    epoch: int
    loss: float
    perplexity: float
    learning_rate: float
    grad_norm: float
    tokens_per_sec: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'session_id': self.session_id,
            'model_id': self.model_id,
            'step': self.step,
            'epoch': self.epoch,
            'loss': round(self.loss, 6),
            'perplexity': round(self.perplexity, 4),
            'learning_rate': self.learning_rate,
            'grad_norm': round(self.grad_norm, 6),
            'tokens_per_sec': round(self.tokens_per_sec, 2),
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class InferenceMetrics:
    """Inference request metrics."""
    model_id: str
    prompt_length: int
    tokens_generated: int
    duration_ms: float
    backend: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_id': self.model_id,
            'prompt_length': self.prompt_length,
            'tokens_generated': self.tokens_generated,
            'duration_ms': round(self.duration_ms, 2),
            'tokens_per_sec': round(self.tokens_generated / (self.duration_ms / 1000), 2) if self.duration_ms > 0 else 0,
            'backend': self.backend,
            'timestamp': self.timestamp.isoformat()
        }


class MetricsCollector:
    """Collects and stores metrics."""
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.training_metrics: List[TrainingMetrics] = []
        self.inference_metrics: List[InferenceMetrics] = []
        self.api_metrics: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()
    
    async def record_training_step(self, metrics: TrainingMetrics) -> None:
        """Record a training step."""
        async with self._lock:
            self.training_metrics.append(metrics)
            if len(self.training_metrics) > self.max_history:
                self.training_metrics = self.training_metrics[-self.max_history:]
        
        # Prometheus
        if PROMETHEUS_AVAILABLE:
            TRAINING_STEP_DURATION.labels(backend='pytorch').observe(
                1.0 / max(metrics.tokens_per_sec, 0.001)
            )
    
    async def record_inference(self, metrics: InferenceMetrics) -> None:
        """Record an inference request."""
        async with self._lock:
            self.inference_metrics.append(metrics)
            if len(self.inference_metrics) > self.max_history:
                self.inference_metrics = self.inference_metrics[-self.max_history:]
        
        # Prometheus
        if PROMETHEUS_AVAILABLE:
            INFERENCE_DURATION.observe(metrics.duration_ms / 1000)
            INFERENCE_REQUESTS.labels(
                model_type=metrics.backend,
                status='success'
            ).inc()
    
    async def record_api_request(self, method: str, endpoint: str,
                                 status: int, duration_ms: float) -> None:
        """Record an API request."""
        metric = {
            'method': method,
            'endpoint': endpoint,
            'status': status,
            'duration_ms': round(duration_ms, 2),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        async with self._lock:
            self.api_metrics.append(metric)
            if len(self.api_metrics) > self.max_history:
                self.api_metrics = self.api_metrics[-self.max_history:]
        
        # Prometheus
        if PROMETHEUS_AVAILABLE:
            status_class = f"{status // 100}xx"
            API_REQUESTS.labels(
                method=method,
                endpoint=endpoint,
                status=status_class
            ).inc()
            API_REQUEST_DURATION.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration_ms / 1000)
    
    async def get_training_summary(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get training summary."""
        async with self._lock:
            metrics = self.training_metrics
            if session_id:
                metrics = [m for m in metrics if m.session_id == session_id]
        
        if not metrics:
            return {'error': 'No metrics found'}
        
        losses = [m.loss for m in metrics]
        tokens_per_sec = [m.tokens_per_sec for m in metrics]
        
        return {
            'session_id': session_id,
            'total_steps': len(metrics),
            'initial_loss': losses[0] if losses else None,
            'final_loss': losses[-1] if losses else None,
            'best_loss': min(losses) if losses else None,
            'avg_tokens_per_sec': sum(tokens_per_sec) / len(tokens_per_sec) if tokens_per_sec else 0,
            'start_time': metrics[0].timestamp.isoformat() if metrics else None,
            'end_time': metrics[-1].timestamp.isoformat() if metrics else None,
        }
    
    async def get_inference_summary(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """Get inference summary."""
        async with self._lock:
            metrics = self.inference_metrics
            if model_id:
                metrics = [m for m in metrics if m.model_id == model_id]
        
        if not metrics:
            return {'error': 'No metrics found'}
        
        durations = [m.duration_ms for m in metrics]
        tokens_generated = [m.tokens_generated for m in metrics]
        
        return {
            'model_id': model_id,
            'total_requests': len(metrics),
            'avg_duration_ms': sum(durations) / len(durations),
            'min_duration_ms': min(durations),
            'max_duration_ms': max(durations),
            'total_tokens_generated': sum(tokens_generated),
            'avg_tokens_per_request': sum(tokens_generated) / len(tokens_generated) if tokens_generated else 0,
        }
    
    async def get_api_summary(self, last_n: int = 1000) -> Dict[str, Any]:
        """Get API summary."""
        async with self._lock:
            metrics = self.api_metrics[-last_n:]
        
        if not metrics:
            return {'error': 'No metrics found'}
        
        status_counts = {}
        endpoint_counts = {}
        total_duration = 0
        
        for m in metrics:
            status = m['status']
            status_counts[status] = status_counts.get(status, 0) + 1
            
            endpoint = m['endpoint']
            endpoint_counts[endpoint] = endpoint_counts.get(endpoint, 0) + 1
            
            total_duration += m['duration_ms']
        
        return {
            'total_requests': len(metrics),
            'avg_duration_ms': round(total_duration / len(metrics), 2),
            'status_distribution': status_counts,
            'endpoint_distribution': endpoint_counts,
        }


# =============================================================================
# Decorators
# =============================================================================

def timed(metric_name: Optional[str] = None):
    """Decorator to time function execution."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.time()
            try:
                return await func(*args, **kwargs)
            finally:
                duration = time.time() - start
                name = metric_name or func.__name__
                if PROMETHEUS_AVAILABLE:
                    # Could add custom histogram here
                    pass
                logger.debug(f"{name} took {duration:.3f}s")
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                duration = time.time() - start
                name = metric_name or func.__name__
                logger.debug(f"{name} took {duration:.3f}s")
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


@contextmanager
def timed_context(name: str):
    """Context manager for timing blocks."""
    start = time.time()
    try:
        yield
    finally:
        duration = time.time() - start
        logger.debug(f"{name} took {duration:.3f}s")


# =============================================================================
# Utility Functions
# =============================================================================

def record_model_creation(backend: str, success: bool = True) -> None:
    """Record model creation metric."""
    if PROMETHEUS_AVAILABLE:
        status = 'success' if success else 'failure'
        MODEL_CREATIONS.labels(backend=backend, status=status).inc()


def record_model_deletion() -> None:
    """Record model deletion metric."""
    if PROMETHEUS_AVAILABLE:
        MODEL_DELETIONS.inc()


def record_training_start(backend: str) -> None:
    """Record training start."""
    if PROMETHEUS_AVAILABLE:
        TRAINING_SESSIONS_STARTED.labels(backend=backend).inc()
        ACTIVE_TRAINING_SESSIONS.labels(backend=backend).inc()


def record_training_end(backend: str, success: bool = True) -> None:
    """Record training end."""
    if PROMETHEUS_AVAILABLE:
        status = 'success' if success else 'failure'
        TRAINING_SESSIONS_COMPLETED.labels(backend=backend, status=status).inc()
        ACTIVE_TRAINING_SESSIONS.labels(backend=backend).dec()


def record_websocket_event(event_type: str) -> None:
    """Record WebSocket event."""
    if PROMETHEUS_AVAILABLE:
        WEBSOCKET_CONNECTIONS.labels(event_type=event_type).inc()


def update_gpu_metrics(device: str, memory_used: float, utilization: float) -> None:
    """Update GPU metrics."""
    if PROMETHEUS_AVAILABLE:
        GPU_MEMORY_USED.labels(device=device).set(memory_used)
        GPU_UTILIZATION.labels(device=device).set(utilization)


def set_active_models(backend: str, count: int) -> None:
    """Set active models gauge."""
    if PROMETHEUS_AVAILABLE:
        ACTIVE_MODELS.labels(backend=backend).set(count)


# Global collector
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def set_metrics_collector(collector: MetricsCollector) -> None:
    """Set global metrics collector."""
    global _metrics_collector
    _metrics_collector = collector
