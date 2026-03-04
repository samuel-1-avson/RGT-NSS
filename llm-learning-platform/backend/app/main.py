"""
Interactive LLM Learning Platform - Ultra-Enhanced Edition
Main FastAPI Application Entry Point
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse

import socketio
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.config import get_settings
from app.database import create_db_and_tables
from app.cache import init_redis, close_redis
from app.api import models, training, inference, tokenization, embeddings, health
from app.api import modules, users, visualizations
from app.api import rlhf, lora, evaluation, inference_opt, interpretability
from app.api import distributed, prompt_eng, safety, long_context
from app.websocket import sio

# ─── Rate Limiter ────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    settings = get_settings()
    print(f"  Starting {settings.app_name} v{settings.app_version}")
    print(f"  Environment: {settings.environment}")
    print(f"  GPU enabled: {settings.use_gpu}")
    # Create database tables on startup
    await create_db_and_tables()
    print("  Database tables ready")
    # Initialize Redis (non-fatal if unavailable)
    redis = await init_redis()
    print(f"  Redis: {'connected' if redis else 'unavailable (caching disabled)'}")
    yield
    await close_redis()
    print("  Shutting down...")


app = FastAPI(
    title="Interactive LLM Learning Platform",
    description=(
        "Ultra-Enhanced educational platform for mastering Large Language Models "
        "from first principles to production deployment."
    ),
    version="3.0.0",
    default_response_class=ORJSONResponse,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─── Rate Limit Error Handler ───────────────────────────────
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ─── CORS Middleware ─────────────────────────────────────────
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Register API Routers ───────────────────────────────────
app.include_router(health.router, tags=["Health"])
app.include_router(models.router, prefix="/api/v1/models", tags=["Models"])
app.include_router(training.router, prefix="/api/v1/training", tags=["Training"])
app.include_router(inference.router, prefix="/api/v1/inference", tags=["Inference"])
app.include_router(tokenization.router, prefix="/api/v1/tokenization", tags=["Tokenization"])
app.include_router(embeddings.router, prefix="/api/v1/embeddings", tags=["Embeddings"])
app.include_router(modules.router, prefix="/api/v1/modules", tags=["Modules"])
app.include_router(users.router, prefix="/api/v1/users", tags=["Users"])
app.include_router(visualizations.router, prefix="/api/v1/visualizations", tags=["Visualizations"])
app.include_router(rlhf.router, prefix="/api/v1/rlhf", tags=["RLHF"])
app.include_router(lora.router, prefix="/api/v1/lora", tags=["LoRA"])
app.include_router(evaluation.router, prefix="/api/v1/evaluation", tags=["Evaluation"])
app.include_router(inference_opt.router, prefix="/api/v1/inference-opt", tags=["Inference Optimization"])
app.include_router(interpretability.router, prefix="/api/v1/interpretability", tags=["Interpretability"])
app.include_router(distributed.router, prefix="/api/v1/distributed", tags=["Distributed Training"])
app.include_router(prompt_eng.router, prefix="/api/v1/prompt-eng", tags=["Prompt Engineering"])
app.include_router(safety.router, prefix="/api/v1/safety", tags=["Safety"])
app.include_router(long_context.router, prefix="/api/v1/long-context", tags=["Long Context"])

# ─── Mount Socket.IO ASGI App ───────────────────────────────
socket_app = socketio.ASGIApp(sio, other_asgi_app=app)
