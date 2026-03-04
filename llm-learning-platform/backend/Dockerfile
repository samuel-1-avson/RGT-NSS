# ──────────────────────────────────────────────────────────────
# Interactive LLM Learning Platform - Backend
# Multi-stage build for production-optimized image
# ──────────────────────────────────────────────────────────────

FROM python:3.11-slim AS base

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ─── Dependencies stage ─────────────────────────────────────
FROM base AS deps

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ─── Production stage ───────────────────────────────────────
FROM base AS runner

# Create non-root user
RUN groupadd --system appuser && \
    useradd --system --gid appuser --create-home appuser

# Copy installed packages from deps stage
COPY --from=deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# Application code
COPY --chown=appuser:appuser . .

# Create necessary directories
RUN mkdir -p checkpoints data logs && \
    chown -R appuser:appuser checkpoints data logs

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:socket_app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
