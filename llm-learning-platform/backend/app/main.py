"""
Interactive LLM Learning Platform - Backend API
Built with FastAPI and custom from-scratch deep learning framework.
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from app.api.routes import router
from app.api.atomic_routes import router as atomic_router
from app.api.compute_routes import router as compute_router


# Application metadata
APP_TITLE = "Interactive LLM Learning Platform API"
APP_DESCRIPTION = """
A comprehensive educational platform for understanding Large Language Models
through interactive visualization and hands-on experimentation.

Built entirely from scratch (no PyTorch/TensorFlow dependencies) to provide
transparent, educational implementations of:
- Custom autograd engine (NumPy-based)
- Atomic GPT implementation (Karpathy-style, pure Python)
- Transformer architecture
- Training loops and optimization
- Tokenization and embeddings

Includes both vectorized (fast) and scalar (educational) implementations
for comparing performance and understanding fundamentals.
"""
APP_VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    print("🚀 Starting Interactive LLM Learning Platform API...")
    print(f"📚 Version: {APP_VERSION}")
    
    # GPU detection
    try:
        import torch
        print(f"🔥 PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"🎮 GPU: {gpu_name} ({vram:.1f} GB VRAM)")
            print(f"⚡ CUDA: {torch.version.cuda}")
        else:
            print("⚠️  CUDA not available — PyTorch will use CPU")
    except ImportError:
        print("📦 PyTorch not installed — using custom NumPy backend only")
    
    # Create necessary directories
    os.makedirs("./checkpoints", exist_ok=True)
    os.makedirs("./data", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    
    yield
    
    # Shutdown
    print("👋 Shutting down API...")


# Create FastAPI application
app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware - Allow all origins for development
# In production, restrict to specific domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=False,  # Must be False when allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)
app.include_router(atomic_router)
app.include_router(compute_router)


@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API information."""
    return f"""
    <!DOCTYPE html>
    <html>
        <head>
            <title>{APP_TITLE}</title>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    max-width: 800px;
                    margin: 50px auto;
                    padding: 20px;
                    line-height: 1.6;
                }}
                h1 {{ color: #333; }}
                .endpoint {{
                    background: #f4f4f4;
                    padding: 10px;
                    border-radius: 5px;
                    margin: 10px 0;
                    font-family: monospace;
                }}
                .method {{
                    display: inline-block;
                    padding: 2px 8px;
                    border-radius: 3px;
                    color: white;
                    font-weight: bold;
                    margin-right: 10px;
                }}
                .get {{ background: #61affe; }}
                .post {{ background: #49cc90; }}
                .ws {{ background: #fca130; }}
            </style>
        </head>
        <body>
            <h1>🧠 {APP_TITLE}</h1>
            <p>{APP_DESCRIPTION.replace(chr(10), '<br>')}</p>
            
            <h2>📚 Documentation</h2>
            <p>
                <a href="/docs">Interactive API Documentation (Swagger UI)</a><br>
                <a href="/redoc">Alternative Documentation (ReDoc)</a>
            </p>
            
            <h2>🔌 Available Endpoints</h2>
            
            <h3>Model Management</h3>
            <div class="endpoint">
                <span class="method post">POST</span> /api/model/create - Create new model
            </div>
            <div class="endpoint">
                <span class="method get">GET</span> /api/model/{{model_id}} - Get model info
            </div>
            <div class="endpoint">
                <span class="method get">GET</span> /api/models - List all models
            </div>
            
            <h3>Training</h3>
            <div class="endpoint">
                <span class="method post">POST</span> /api/training/start - Start training
            </div>
            <div class="endpoint">
                <span class="method get">GET</span> /api/training/{{session_id}}/status - Training status
            </div>
            
            <h3>Inference</h3>
            <div class="endpoint">
                <span class="method post">POST</span> /api/inference/generate - Generate text
            </div>
            <div class="endpoint">
                <span class="method post">POST</span> /api/inference/tokenize - Tokenize text
            </div>
            
            <h3>Visualization</h3>
            <div class="endpoint">
                <span class="method get">GET</span> /api/viz/attention/{{model_id}} - Get attention data
            </div>
            <div class="endpoint">
                <span class="method get">GET</span> /api/viz/embeddings/{{model_id}} - Get embeddings
            </div>
            
            <h3>Atomic GPT (Educational)</h3>
            <div class="endpoint">
                <span class="method post">POST</span> /api/atomic/compute/step - Step-by-step computation
            </div>
            <div class="endpoint">
                <span class="method post">POST</span> /api/atomic/model/create - Create atomic model
            </div>
            <div class="endpoint">
                <span class="method get">GET</span> /api/atomic/demo/gradient_flow - Gradient flow demo
            </div>
            
            <h3>WebSocket</h3>
            <div class="endpoint">
                <span class="method ws">WS</span> /api/ws/training/{{session_id}} - Real-time training updates
            </div>
            
            <h2>🚀 Quick Start</h2>
            <ol>
                <li>Create a model: <code>POST /api/model/create</code></li>
                <li>Start training: <code>POST /api/training/start</code></li>
                <li>Connect via WebSocket for real-time updates</li>
                <li>Generate text: <code>POST /api/inference/generate</code></li>
            </ol>
            
            <p><strong>Version:</strong> {APP_VERSION}</p>
        </body>
    </html>
    """


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": APP_VERSION,
        "components": {
            "api": "up",
            "models": "ready",
            "training": "ready"
        }
    }


@app.get("/api/status")
async def api_status():
    """Get API status and statistics."""
    from app.api.routes import _models, _training_sessions
    
    return {
        "version": APP_VERSION,
        "active_models": len(_models),
        "active_training_sessions": len(_training_sessions),
        "endpoints": {
            "total": len(app.routes),
            "documented": len([r for r in app.routes if hasattr(r, 'methods')])
        }
    }


# Mount static files if they exist
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    reload = os.environ.get("ENVIRONMENT") == "development"
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )
