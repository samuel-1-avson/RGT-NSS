"""
Application configuration using pydantic-settings.
"""

import secrets
from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ─── Application ─────────────────────────────────────────
    app_name: str = "Interactive LLM Learning Platform"
    app_version: str = "3.0.0"
    environment: str = "development"
    debug: bool = False
    log_level: str = "info"

    # ─── Server ──────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000

    # ─── CORS ────────────────────────────────────────────────
    cors_origins: List[str] = ["http://localhost:3000"]

    # ─── Database ────────────────────────────────────────────
    database_url: str = "postgresql+asyncpg://llmuser:llmpass@localhost:5432/llmplatform"

    # ─── Redis ───────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"

    # ─── Security ────────────────────────────────────────────
    # Auto-generated per startup if SECRET_KEY env var is not set.
    # In production, always set SECRET_KEY explicitly.
    secret_key: str = secrets.token_urlsafe(64)
    access_token_expire_minutes: int = 1440  # 24 hours

    # ─── GPU ─────────────────────────────────────────────────
    use_gpu: bool = False
    cuda_visible_devices: str = "0"

    # ─── Storage ─────────────────────────────────────────────
    checkpoint_dir: str = "checkpoints"
    data_dir: str = "data"
    max_upload_size_mb: int = 100

    # ─── Training Limits ─────────────────────────────────────
    max_concurrent_training_sessions: int = 10
    max_training_steps: int = 10000
    max_model_parameters: int = 50_000_000  # 50M

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache()
def get_settings() -> Settings:
    return Settings()
