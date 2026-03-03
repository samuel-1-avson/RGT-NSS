# Interactive LLM Learning Platform

A comprehensive, web-based educational platform for understanding Large Language Models through interactive visualization and hands-on experimentation.

## 🎯 Vision

This platform enables learners to:
- **Build** a GPT-style transformer from scratch
- **Visualize** every computational step in real-time
- **Experiment** with hyperparameters and observe immediate effects
- **Understand** the mathematical foundations through interactive examples
- **Train** micro-models in the browser with live feedback

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Frontend (Next.js + React)                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Interactive │  │  Real-time   │  │  Educational │          │
│  │  Components  │  │  Charts      │  │  Modules     │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ HTTP / WebSocket
┌─────────────────────────────────────────────────────────────────┐
│                     Backend (FastAPI + Python)                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Custom Deep Learning Framework              │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐  │    │
│  │  │ Autograd │  │  GPT     │  │ Training │  │Optimizers│ │    │
│  │  │ Engine   │  │  Model   │  │ Engine   │  │          │ │    │
│  │  └──────────┘  └──────────┘  └──────────┘  └────────┘  │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## ✨ Features

### Educational Modules
1. **🔤 Tokenization Lab** - Visualize text-to-tokens conversion
2. **🎯 Embedding Explorer** - Explore vector representations
3. **🔍 Attention Visualizer** - Interactive attention heatmaps
4. **🏗️ Transformer Block** - Step-by-step architecture breakdown
5. **🚂 Training Dashboard** - Real-time training visualization
6. **💬 Inference Playground** - Text generation experimentation

### Core Capabilities
- **From-Scratch Implementation**: No PyTorch/TensorFlow - pure NumPy
- **Real-time Training**: Live loss curves, metrics, and visualizations
- **Interactive Visualizations**: D3.js-powered attention matrices, embedding spaces
- **Model Presets**: Nano (100K) to Medium (100M) parameters
- **WebSocket Streaming**: Instant training updates

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 20+
- Docker (optional)

### Development Setup

#### 1. Clone and Setup
```bash
git clone <repository>
cd llm-learning-platform
```

#### 2. Backend Setup
```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --reload
```

Backend will be available at `http://localhost:8000`

#### 3. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

Frontend will be available at `http://localhost:3000`

### Docker Setup (Production)
```bash
docker-compose up --build
```

Access the application at `http://localhost:3000`

## 📁 Project Structure

```
llm-learning-platform/
├── backend/
│   ├── app/
│   │   ├── core/
│   │   │   ├── tensor.py          # Autograd engine
│   │   │   ├── module.py          # Neural network base classes
│   │   │   ├── optimizer.py       # SGD, Adam, AdamW
│   │   │   └── trainer.py         # Training engine
│   │   ├── models/
│   │   │   └── gpt.py             # GPT model implementation
│   │   ├── api/
│   │   │   └── routes.py          # REST & WebSocket endpoints
│   │   └── main.py                # FastAPI application
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/
│   ├── app/                       # Next.js app router
│   │   ├── learn/
│   │   │   ├── tokenization/
│   │   │   ├── embeddings/
│   │   │   ├── attention/
│   │   │   └── ...
│   │   ├── train/
│   │   └── page.tsx               # Landing page
│   ├── components/
│   ├── stores/
│   │   └── modelStore.ts          # Zustand state management
│   ├── utils/
│   │   └── api.ts                 # API client & WebSocket
│   ├── package.json
│   └── Dockerfile
│
├── docker-compose.yml
└── README.md
```

## 🔬 Technical Deep Dive

### Custom Autograd Engine

Built from scratch for educational transparency:

```python
class Tensor:
    """Tensor with automatic differentiation"""
    def __init__(self, data, requires_grad=True):
        self.data = np.array(data)
        self.grad = np.zeros_like(data)
        self._backward = lambda: None
    
    def backward(self):
        """Reverse-mode autodiff"""
        # Topological sort + backprop
        ...
```

### GPT Model Architecture

Complete transformer implementation:

```python
class MicroGPT(Module):
    """Educational GPT model"""
    
    def __init__(self, config: GPTConfig):
        self.token_emb = Embedding(config.vocab_size, config.d_model)
        self.pos_emb = Embedding(config.max_seq_len, config.d_model)
        self.blocks = [TransformerBlock(config) for _ in range(config.num_layers)]
        self.norm_f = RMSNorm(config.d_model)
        self.lm_head = Linear(config.d_model, config.vocab_size)
```

### Key Components

1. **Multi-Head Attention**: Self-attention with Q, K, V projections
2. **RMSNorm**: Modern normalization (used in Llama)
3. **SwiGLU Activation**: Advanced feedforward activation
4. **Causal Masking**: Autoregressive generation support

## 🎓 Learning Paths

### Beginner Path
1. Tokenization & Embeddings (Week 1)
2. Attention Mechanisms (Week 2)
3. Transformer Architecture (Week 3)
4. Training Fundamentals (Week 4)
5. Inference & Generation (Week 5)

### Intermediate Path
1. Attention Deep Dive
2. Transformer Blocks
3. Training & Optimization
4. Inference & Generation

### Expert Path
1. Advanced Attention Patterns
2. Training Optimization
3. Model Efficiency
4. Research Frontiers

## 📊 API Endpoints

### Model Management
- `POST /api/model/create` - Create new model
- `GET /api/model/{id}` - Get model info
- `POST /api/model/{id}/reset` - Reset parameters

### Training
- `POST /api/training/start` - Start training session
- `POST /api/training/{id}/stop` - Stop training
- `GET /api/training/{id}/status` - Get status
- `WS /api/ws/training/{id}` - Real-time updates

### Inference
- `POST /api/inference/generate` - Generate text
- `POST /api/inference/tokenize` - Tokenize text

### Visualization
- `GET /api/viz/attention/{id}` - Get attention weights
- `GET /api/viz/embeddings/{id}` - Get embedding projections

## 🛠️ Development

### Adding a New Module

1. Create page at `frontend/app/learn/{module}/page.tsx`
2. Add API endpoint in `backend/app/api/routes.py`
3. Update navigation in `frontend/app/learn/page.tsx`

### Adding a Visualization

1. Create D3/React component in `frontend/components/`
2. Add data endpoint in backend
3. Connect via API client

## 🧪 Testing

### Backend Tests
```bash
cd backend
pytest tests/
```

### Frontend Tests
```bash
cd frontend
npm test
```

## 📈 Performance Benchmarks

| Model Size | Parameters | Training Speed | Memory |
|------------|-----------|----------------|--------|
| Nano | 100K | ~1000 tok/s | 2 MB |
| Micro | 1M | ~500 tok/s | 8 MB |
| Small | 10M | ~100 tok/s | 80 MB |
| Medium | 100M | ~10 tok/s | 400 MB |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📚 Resources

### Recommended Reading
- "Attention Is All You Need" (Vaswani et al., 2017)
- "The Illustrated Transformer" (Jay Alammar)
- "Let's Build GPT" (Andrej Karpathy)

### Video Resources
- Andrej Karpathy's Neural Networks: Zero to Hero
- 3Blue1Brown Neural Network Series

## 📝 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- Built for educational purposes
- Inspired by Andrej Karpathy's educational content
- No external ML frameworks - everything from scratch

---

**Happy Learning!** 🎓🤖
