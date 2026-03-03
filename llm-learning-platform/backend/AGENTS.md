# LLM Learning Platform Backend - AGENTS.md

**Version:** 2.0.0  
**Last Updated:** March 3, 2026  
**Status:** Production-Ready

---

## 📋 Overview

This document contains essential information for AI agents working on the LLM Learning Platform backend. It describes the architecture, critical improvements made, and best practices for development.

---

## 🏗️ Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Backend Architecture                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐ │
│  │   FastAPI    │   │     Auth     │   │   Schemas    │   │   Metrics    │ │
│  │    Router    │──▶│   System     │   │ Validation   │   │ Monitoring   │ │
│  └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘ │
│         │                                                                      │
│         ▼                                                                      │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    State Manager (Persistent Storage)                 │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │   │
│  │  │   Memory     │  │    Redis     │  │    File      │              │   │
│  │  │  (dev/test)  │  │ (production) │  │  (backup)    │              │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘              │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│         │                                                                      │
│         ▼                                                                      │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         Core Framework                                │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │   │
│  │  │   Tensor     │  │    Module    │  │  Optimizer   │              │   │
│  │  │  (Autograd)  │  │   (Layers)   │  │ (SGD/Adam)   │              │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘              │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │   │
│  │  │  Tokenizer   │  │    Trainer   │  │  Exceptions  │              │   │
│  │  │(Char/BPE/etc)│  │   (Engine)   │  │  (Custom)    │              │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘              │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│         │                                                                      │
│         ▼                                                                      │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         Model Backends                                │   │
│  │  ┌──────────────────────────┐  ┌──────────────────────────┐         │   │
│  │  │      MicroGPT            │  │      PyTorchGPT          │         │   │
│  │  │   (NumPy/Custom)         │  │   (GPU-Accelerated)      │         │   │
│  │  │   - Educational          │  │   - Production           │         │   │
│  │  │   - CPU Only             │  │   - CUDA/FlashAttention  │         │   │
│  │  │   - From Scratch         │  │   - KV-Cache             │         │   │
│  │  └──────────────────────────┘  └──────────────────────────┘         │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Critical Improvements Implemented

### 1. Fixed Import Structure

**Before:**
```python
import sys
sys.path.append('..')  # ❌ Fragile
from app.models.gpt import MicroGPT
```

**After:**
```python
from app.models.gpt import MicroGPT  # ✅ Clean absolute import
```

**Files Modified:**
- `app/api/routes.py`
- `app/core/trainer.py`
- `app/models/gpt.py`
- `app/api/atomic_routes.py`
- `app/api/compute_routes.py`

### 2. Added Persistent State Management

**Location:** `app/core/state_manager.py`

**Features:**
- In-memory storage for development
- Redis storage for production
- Automatic TTL management
- Health checks

**Usage:**
```python
from app.core.state_manager import get_state_manager, RedisStateManager

# Development
manager = get_state_manager()  # MemoryStateManager

# Production
redis_manager = RedisStateManager(redis_url="redis://localhost:6379")
set_state_manager(redis_manager)

# Use
await manager.save_model(model_id, model, metadata={"name": "My Model"})
model = await manager.get_model(model_id)
```

### 3. Implemented Authentication System

**Location:** `app/core/auth.py`

**Features:**
- JWT token authentication
- API key authentication
- Rate limiting
- Permission system

**Usage:**
```python
from app.core.auth import get_current_user, require_permissions
from fastapi import Depends

@router.post("/model/create")
async def create_model(
    config: GPTConfigRequest,
    user: User = Depends(get_current_user)
):
    # User is authenticated
    pass

@router.delete("/model/{id}")
async def delete_model(
    model_id: str,
    user: User = Depends(require_permissions("write", "delete"))
):
    # User has required permissions
    pass
```

**Default API Key (Development):**
```
llm_dev_admin_key_change_in_production
```

### 4. Created Comprehensive Exception Handling

**Location:** `app/core/exceptions.py`

**Exception Types:**
- `ModelNotFoundError`
- `ModelCreationError`
- `TrainingSessionNotFoundError`
- `TrainingError`
- `InferenceError`
- `AuthenticationError`
- `AuthorizationError`
- `ValidationError`
- `StorageError`
- `GPUError`

**Usage:**
```python
from app.core.exceptions import ModelNotFoundError

@router.get("/model/{model_id}")
async def get_model(model_id: str):
    if model_id not in _models:
        raise ModelNotFoundError(model_id)
    return _models[model_id]
```

### 5. Implemented Proper Tokenizers

**Location:** `app/core/tokenizer.py`

**Available Tokenizers:**
- `CharacterTokenizer` - Character-level
- `WordTokenizer` - Word-level
- `BPETokenizer` - Byte-Pair Encoding
- `TiktokenTokenizer` - OpenAI's tiktoken

**Usage:**
```python
from app.core.tokenizer import TokenizerFactory, get_default_tokenizer

# Create tokenizer
tokenizer = TokenizerFactory.create("bpe", vocab_size=10000)
tokenizer.train(corpus_texts)

# Use tokenizer
tokens = tokenizer.encode("Hello, world!")
text = tokenizer.decode(tokens)

# Or use default
tokenizer = get_default_tokenizer()  # CharacterTokenizer
```

### 6. Added Pydantic Schemas

**Location:** `app/schemas/`

**Schema Categories:**
- `models.py` - Model creation and info
- `training.py` - Training configuration and status
- `inference.py` - Generation and tokenization
- `common.py` - Shared schemas

**Benefits:**
- Automatic validation
- Clear API contracts
- Generated documentation
- Type safety

### 7. Improved WebSocket Manager

**Location:** `app/core/websocket_manager.py`

**Features:**
- Connection pooling
- Automatic cleanup
- Health monitoring
- Message queuing

**Usage:**
```python
from app.core.websocket_manager import get_connection_manager

manager = get_connection_manager()

@router.websocket("/ws/training/{session_id}")
async def training_websocket(websocket: WebSocket, session_id: str):
    await manager.connect(websocket, session_id)
    try:
        while True:
            message = await manager.receive_json(websocket)
            # Process message
    except WebSocketDisconnect:
        await manager.disconnect(websocket, session_id)
```

### 8. Added Monitoring & Metrics

**Location:** `app/core/metrics.py`

**Features:**
- Prometheus-compatible metrics
- Training metrics collection
- Inference metrics
- API request tracking

**Metrics Available:**
- `llm_models_created_total`
- `llm_training_sessions_started_total`
- `llm_inference_requests_total`
- `llm_api_request_duration_seconds`
- `llm_gpu_memory_used_bytes`

**Usage:**
```python
from app.core.metrics import timed, get_metrics_collector

@timed("model_creation")
async def create_model(config):
    # Function is automatically timed
    pass

# Record custom metrics
collector = get_metrics_collector()
await collector.record_training_step(metrics)
```

### 9. Created Comprehensive Tests

**Location:** `tests/`

**Test Files:**
- `test_models.py` - Model implementation tests
- `test_tokenizer.py` - Tokenizer tests
- `test_state_manager.py` - State management tests
- `test_api.py` - API endpoint tests

**Running Tests:**
```bash
cd backend
pytest tests/ -v --cov=app --cov-report=html
```

### 10. Updated Dependencies

**Location:** `requirements.txt`

**Key Additions:**
```
# Authentication
python-jose[cryptography]>=3.3.0
passlib[bcrypt]>=1.7.4

# Tokenization
tiktoken>=0.6.0
tokenizers>=0.15.0

# State Management
redis>=5.0.0
aioredis>=2.0.0

# Monitoring
prometheus-client>=0.19.0
opentelemetry-api>=1.22.0

# Background Tasks
celery>=5.3.0

# Testing
pytest>=8.0.0
pytest-asyncio>=0.23.0
pytest-cov>=4.1.0
```

---

## 📁 File Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py                      # FastAPI entry point
│   │
│   ├── api/                         # API routes
│   │   ├── __init__.py
│   │   ├── routes.py                # Main API routes
│   │   ├── atomic_routes.py         # Educational routes
│   │   └── compute_routes.py        # Computation demos
│   │
│   ├── core/                        # Core framework
│   │   ├── __init__.py
│   │   ├── tensor.py                # Autograd engine
│   │   ├── module.py                # Neural network layers
│   │   ├── optimizer.py             # Optimization algorithms
│   │   ├── trainer.py               # Training engine
│   │   ├── pytorch_trainer.py       # PyTorch trainer
│   │   ├── atomic_gpt.py            # Educational GPT
│   │   ├── exceptions.py            # ✅ Custom exceptions
│   │   ├── state_manager.py         # ✅ Persistent storage
│   │   ├── auth.py                  # ✅ Authentication
│   │   ├── tokenizer.py             # ✅ Tokenizers
│   │   ├── websocket_manager.py     # ✅ WebSocket handling
│   │   └── metrics.py               # ✅ Monitoring
│   │
│   ├── models/                      # Model implementations
│   │   ├── __init__.py
│   │   ├── gpt.py                   # MicroGPT (NumPy)
│   │   └── pytorch_gpt.py           # PyTorchGPT (GPU)
│   │
│   ├── schemas/                     # ✅ Pydantic schemas
│   │   ├── __init__.py
│   │   ├── models.py
│   │   ├── training.py
│   │   ├── inference.py
│   │   └── common.py
│   │
│   └── utils/                       # Utilities
│       └── __init__.py
│
├── tests/                           # ✅ Test suite
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_models.py
│   ├── test_tokenizer.py
│   ├── test_state_manager.py
│   └── test_api.py
│
├── checkpoints/                     # Model checkpoints
├── data/                           # Training data
├── logs/                           # Application logs
├── requirements.txt                # ✅ Updated dependencies
├── Dockerfile
└── AGENTS.md                       # This file
```

---

## 🔧 Development Guidelines

### Code Style

- Use **Black** for formatting: `black app/ tests/`
- Use **Ruff** for linting: `ruff check app/ tests/`
- Use **mypy** for type checking: `mypy app/`

### Adding New Features

1. **Add Schema First:** Define Pydantic schemas in `app/schemas/`
2. **Implement Core Logic:** Add to `app/core/` or `app/models/`
3. **Create API Route:** Add endpoint in `app/api/`
4. **Write Tests:** Add tests in `tests/`
5. **Update Documentation:** Update this file

### Error Handling

Always use custom exceptions:

```python
# ❌ Don't do this
raise HTTPException(status_code=404, detail="Model not found")

# ✅ Do this
from app.core.exceptions import ModelNotFoundError
raise ModelNotFoundError(model_id)
```

### Authentication

Always use authentication dependencies:

```python
from app.core.auth import get_current_user

@router.post("/endpoint")
async def endpoint(user: User = Depends(get_current_user)):
    # User is guaranteed to be authenticated
    pass
```

### State Management

Always use the state manager:

```python
from app.core.state_manager import get_state_manager

manager = get_state_manager()
await manager.save_model(model_id, model, metadata)
```

---

## 🚀 Deployment

### Environment Variables

```bash
# Required
JWT_SECRET_KEY="your-secret-key-change-in-production"

# Optional
REDIS_URL="redis://localhost:6379"
ENVIRONMENT="production"
LOG_LEVEL="info"

# Rate Limiting
RATE_LIMIT_REQUESTS="100"
RATE_LIMIT_WINDOW="60"

# Training
DEFAULT_CHECKPOINT_DIR="./checkpoints"
DEFAULT_DATA_DIR="./data"
```

### Docker Deployment

```bash
# Build
docker build -t llm-platform-backend .

# Run
docker run -p 8000:8000 \
  -e JWT_SECRET_KEY="secret" \
  -e REDIS_URL="redis://host:6379" \
  llm-platform-backend
```

### Docker Compose

```bash
docker-compose up -d
```

---

## 📊 Monitoring

### Prometheus Metrics

Metrics are exposed at `/metrics`:

```
llm_models_created_total{backend="pytorch",status="success"} 10
llm_active_training_sessions{backend="pytorch"} 2
llm_api_request_duration_seconds_bucket{method="POST",endpoint="/api/model/create",le="0.1"} 5
```

### Health Checks

- `/health` - Basic health check
- `/api/status` - Detailed status with metrics

---

## 🧪 Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Run with Coverage

```bash
pytest tests/ --cov=app --cov-report=html
```

### Run Specific Test

```bash
pytest tests/test_models.py::TestMicroGPT::test_forward_pass -v
```

---

## 🐛 Troubleshooting

### Common Issues

**Import Errors:**
```bash
# Make sure you're in the backend directory
cd backend
python -m pytest tests/
```

**Redis Connection:**
```bash
# Check Redis is running
redis-cli ping

# Or use memory backend for testing
STATE_BACKEND=memory
```

**GPU Not Available:**
```bash
# Check PyTorch installation
python -c "import torch; print(torch.cuda.is_available())"

# Use CPU backend
backend=custom
```

---

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Redis Documentation](https://redis.io/documentation)
- [Prometheus Documentation](https://prometheus.io/docs/)

---

## 📝 Changelog

### Version 2.0.0 (March 3, 2026)

**Major Improvements:**
- ✅ Fixed import structure (removed sys.path.append)
- ✅ Added persistent state management (Redis)
- ✅ Implemented authentication system (JWT + API keys)
- ✅ Created comprehensive exception handling
- ✅ Added proper tokenizers (BPE, tiktoken)
- ✅ Implemented Pydantic schemas for validation
- ✅ Improved WebSocket reliability
- ✅ Added monitoring and metrics (Prometheus)
- ✅ Created comprehensive test suite
- ✅ Updated all dependencies

**Production Ready:**
- Authentication and authorization
- Persistent storage
- Rate limiting
- Error handling
- Monitoring
- Comprehensive tests

---

**Maintained by:** AI Development Team  
**For questions:** Refer to analysis_report.md for detailed architecture information
