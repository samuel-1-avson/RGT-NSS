# Deployment Guide

## Architecture Overview

```
┌─────────────────────┐        ┌──────────────────────────────┐
│   Vercel (Frontend)  │ ──→   │  Docker Host (Backend Stack)  │
│   Next.js 15 SSR     │  API  │  ┌─────────────────────────┐  │
│   Static + Edge      │ calls │  │ FastAPI Backend (:8000)  │  │
└─────────────────────┘        │  └────────┬────────────────┘  │
                               │           │                    │
                               │  ┌────────┴───────┐           │
                               │  │  PostgreSQL 16  │           │
                               │  │  Redis 7        │           │
                               │  └────────────────┘           │
                               └──────────────────────────────┘
```

---

## Frontend → Vercel

### 1. Prerequisites
- [Vercel account](https://vercel.com/signup)
- GitHub / GitLab repository connected

### 2. Deploy

**Option A — Vercel CLI:**
```bash
cd frontend
npx vercel --prod
```

**Option B — Vercel Dashboard:**
1. Go to [vercel.com/new](https://vercel.com/new)
2. Import your repository
3. Set **Root Directory** to `llm-learning-platform/frontend`
4. Framework Preset: **Next.js** (auto-detected)
5. Click **Deploy**

### 3. Environment Variables

Set these in **Vercel Dashboard → Project → Settings → Environment Variables**:

| Variable | Value | Environment |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | `https://api.your-domain.com` | Production |
| `NEXT_PUBLIC_WS_URL` | `wss://api.your-domain.com` | Production |

### 4. Custom Domain (optional)
1. Go to **Settings → Domains**
2. Add your domain
3. Update DNS records as instructed

---

## Backend → Docker

### 1. Prerequisites
- Docker Engine 24+
- Docker Compose v2+
- Server with 2+ GB RAM

### 2. Quick Start (Development)

```bash
cd llm-learning-platform

# Copy environment file
cp .env.example .env

# Start all services
docker compose up -d

# Check status
docker compose ps

# View logs
docker compose logs -f backend
```

### 3. Production Deployment

```bash
# Update .env for production
# - Set ENVIRONMENT=production
# - Set strong POSTGRES_PASSWORD
# - Set CORS_ORIGINS to your Vercel frontend URL

# Build and start
docker compose up -d --build

# Scale backend (if needed)
docker compose up -d --scale backend=3
```

### 4. GPU Support

```bash
# Requires NVIDIA Container Toolkit
docker compose -f docker-compose.gpu.yml up -d
```

### 5. Services

| Service | Port | Description |
|---|---|---|
| Backend API | `:8000` | FastAPI with 18 routers |
| PostgreSQL | `:5432` | Primary database |
| Redis | `:6379` | Caching & sessions |
| Traefik | `:80` / `:8080` | Reverse proxy & dashboard |

### 6. API Documentation

Once running, access:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **Health Check**: `http://localhost:8000/health`

### 7. API Endpoints (18 Routers)

**Core Modules:**
- `/api/v1/models` — Model architecture
- `/api/v1/training` — Training engine
- `/api/v1/inference` — Inference engine
- `/api/v1/tokenization` — Tokenization (BPE, WordPiece, Unigram)
- `/api/v1/embeddings` — Embedding engine
- `/api/v1/modules` — Module management
- `/api/v1/users` — User management
- `/api/v1/visualizations` — Visualization data

**Advanced Modules:**
- `/api/v1/rlhf` — Reward modeling, PPO, DPO
- `/api/v1/lora` — LoRA, QLoRA, PEFT
- `/api/v1/evaluation` — BLEU, ROUGE, perplexity, benchmarks

**Frontier Modules:**
- `/api/v1/inference-opt` — KV cache, quantization, speculative decoding
- `/api/v1/interpretability` — Logit lens, activation patching, circuits
- `/api/v1/distributed` — Data/model/pipeline parallelism, ZeRO
- `/api/v1/prompt-eng` — Templates, analysis, comparison
- `/api/v1/safety` — Evaluation, red teaming, constitutional AI
- `/api/v1/long-context` — RoPE scaling, ALiBi, position methods

---

## Running Tests

```bash
cd llm-learning-platform/backend

# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_core_engines.py -v

# Run specific test class
pytest tests/test_api.py::TestRLHFAPI -v
```

---

## Production Checklist

- [ ] Set strong database credentials in `.env`
- [ ] Set `ENVIRONMENT=production`
- [ ] Set `CORS_ORIGINS` to your Vercel frontend URL only
- [ ] Configure SSL/TLS (via Traefik or reverse proxy)
- [ ] Set up database backups for PostgreSQL
- [ ] Configure monitoring (Prometheus metrics at `/metrics`)
- [ ] Set `LOG_LEVEL=warning` for production
- [ ] Enable Redis persistence (`appendonly yes` already configured)
