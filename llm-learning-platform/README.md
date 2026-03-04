# Interactive LLM Learning Platform - Ultra-Enhanced Edition

## Comprehensive Educational Environment for Mastering Large Language Models

The **Interactive LLM Learning Platform** is a full-stack educational platform for learning LLMs from first principles to production deployment. Build GPT-style transformers from scratch, visualize computations in real-time, experiment with hyperparameters, and master advanced topics like RLHF, LoRA, and inference optimization.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│  Frontend (Next.js 15 + React 18 + TypeScript)           │
│  • Interactive Modules (Tokenization, Attention, etc.)   │
│  • Visualization Engine (D3.js / Three.js / Recharts)    │
│  • Real-time Updates (Socket.IO)                         │
│  • State Management (Zustand + React Query)              │
└──────────────────────┬───────────────────────────────────┘
                       │ HTTP / WebSocket
┌──────────────────────┴───────────────────────────────────┐
│  Backend (FastAPI + Python 3.11+)                        │
│  • REST API (19 routers, auto-docs at /docs)             │
│  • Custom Core Engine (Tensor Ops, Autograd, Models)     │
│  • WebSocket Layer (python-socketio)                     │
│  • Rate Limiting (SlowAPI)                               │
└──────────────────────┬───────────────────────────────────┘
                       │
┌──────────────────────┴───────────────────────────────────┐
│  Data Layer                                               │
│  • PostgreSQL 16 (Primary DB via SQLModel/asyncpg)        │
│  • Redis 7 (Cache / Rate Limiting)                        │
│  • Alembic (Database Migrations)                          │
└──────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 20+
- Docker & Docker Compose
- (Optional) NVIDIA GPU with CUDA 12+

### Development Setup

```bash
# Clone and setup
git clone <repo-url>
cd llm-learning-platform

# Copy environment configuration
cp .env.example .env
# ⚠️ Edit .env and set a strong SECRET_KEY for production

# Backend
cd backend
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Frontend (separate terminal)
cd frontend
npm install
npm run dev
```

### Docker Setup

```bash
# Standard (CPU)
docker-compose up --build

# With GPU support
docker-compose -f docker-compose.gpu.yml up --build
```

### Running Tests

```bash
# Backend tests
cd backend
python -m pytest tests/ -v

# Frontend tests
cd frontend
npm test
```

## Platform Modules

| # | Module | Status | Description |
|---|--------|--------|-------------|
| 1 | Tokenization Laboratory | ✅ | BPE, WordPiece, SentencePiece from scratch |
| 2 | Embedding Explorer | ✅ | Positional encodings, similarity, analogy |
| 3 | Attention Mechanism Visualizer | ✅ | Full/local/sparse/linear attention |
| 4 | Transformer Block Breakdown | ✅ | Norm, MLP, residual connections |
| 5 | Training Dashboard | ✅ | Loss curves, gradient flow, optimization |
| 6 | Inference Playground | ✅ | Generation with sampling strategies |
| 7 | RLHF & Constitutional AI Lab | ✅ | Reward modeling, PPO, DPO |
| 8 | Parameter-Efficient Fine-tuning | ✅ | LoRA, QLoRA, Adapters |
| 9 | Distributed Training Simulator | ✅ | Data/Model parallelism, ZeRO |
| 10 | Inference Optimization Lab | ✅ | KV cache, speculative decoding |
| 11 | Model Evaluation & Benchmarking | ✅ | BLEU, ROUGE, perplexity |
| 12 | Prompt Engineering Workshop | ✅ | Advanced prompting techniques |
| 13 | Mechanistic Interpretability | ✅ | Logit lens, neuron analysis |
| 14 | Long Context Techniques | ✅ | RoPE, YaRN, ALiBi |
| 15 | AI Safety & Alignment Center | ✅ | Red-teaming, bias detection |
| 16 | Quantization Lab | ✅ | INT8, INT4, NF4 quantization |

## Tech Stack

- **Frontend**: Next.js 15, React 18, TypeScript 5.3, Tailwind CSS, D3.js, Three.js, Zustand, React Query
- **Backend**: FastAPI, Python 3.11+, NumPy, WebSocket (python-socketio)
- **Database**: PostgreSQL 16 (async via asyncpg), Redis 7+
- **Infrastructure**: Docker, Traefik, Alembic (migrations)
- **Testing**: Pytest (backend), Vitest + React Testing Library (frontend)
- **CI/CD**: GitHub Actions

## License

MIT
