# Interactive LLM Learning Platform - Ultra-Enhanced Edition

## Comprehensive Educational Environment for Mastering Large Language Models

The **Interactive LLM Learning Platform** is a full-stack educational platform for learning LLMs from first principles to production deployment. Build GPT-style transformers from scratch, visualize computations in real-time, experiment with hyperparameters, and master advanced topics like RLHF, LoRA, and inference optimization.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│  Frontend (Next.js 15 + React 18 + TypeScript)           │
│  • Interactive Modules (Tokenization, Attention, etc.)   │
│  • Visualization Engine (D3.js / Three.js)               │
│  • Real-time Collaboration (WebSocket + WebRTC)          │
│  • State Management (Zustand + React Query)              │
└──────────────────────┬───────────────────────────────────┘
                       │ HTTP / WebSocket
┌──────────────────────┴───────────────────────────────────┐
│  Backend (FastAPI + Python 3.11+)                        │
│  • REST API + WebSocket + GraphQL                        │
│  • Microservices (Model, Training, Inference, etc.)      │
│  • Core Engine (Tensor Ops, Autograd, Models)            │
│  • Event-Driven Communication (Redis Pub/Sub)            │
└──────────────────────┬───────────────────────────────────┘
                       │
┌──────────────────────┴───────────────────────────────────┐
│  Data Layer                                               │
│  • PostgreSQL (Primary DB)                                │
│  • Redis (Cache / Sessions / Pub/Sub)                     │
│  • MinIO / S3 (Object Storage)                            │
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

## Platform Modules

| # | Module | Description |
|---|--------|-------------|
| 1 | Tokenization Laboratory | BPE, WordPiece, SentencePiece from scratch |
| 2 | Embedding Explorer | Positional encodings, similarity, analogy |
| 3 | Attention Mechanism Visualizer | Full/local/sparse/linear attention |
| 4 | Transformer Block Breakdown | Norm, MLP, residual connections |
| 5 | Training Dashboard | Loss curves, gradient flow, optimization |
| 6 | Model Configuration Studio | Architecture design and comparison |
| 7 | Inference Playground | Generation with sampling strategies |
| 8 | RLHF & Constitutional AI Lab | Reward modeling, PPO, DPO |
| 9 | Parameter-Efficient Fine-tuning | LoRA, QLoRA, Adapters |
| 10 | Distributed Training Simulator | Data/Model parallelism, ZeRO |
| 11 | Inference Optimization Lab | KV cache, speculative decoding |
| 12 | Model Evaluation & Benchmarking | Metrics, leaderboards |
| 13 | Prompt Engineering Workshop | Advanced prompting techniques |
| 14 | Mechanistic Interpretability | Circuit tracing, feature viz |
| 15 | Long Context Techniques | RoPE, YaRN, ALiBi, Ring Attention |
| 16 | AI Safety & Alignment Center | Red-teaming, bias detection |
| 17 | Model Merging & Ensemble Studio | SLERP, task arithmetic |
| 18 | Data Curation Pipeline | Dataset preparation, cleaning |
| 19 | Multimodal Integration | Vision encoders, CLIP |
| 20 | Quantization Lab | INT8, INT4, GPTQ, AWQ |

## Tech Stack

- **Frontend**: Next.js 15, React 18, TypeScript 5.3, Tailwind CSS, D3.js, Three.js, Zustand, React Query
- **Backend**: FastAPI, Python 3.11+, NumPy, WebSocket (python-socketio)
- **Database**: PostgreSQL 16, Redis 7+
- **Infrastructure**: Docker, Traefik, Prometheus, Grafana
- **ML/AI**: PyTorch, HuggingFace Transformers, PEFT, TRL

## License

MIT
