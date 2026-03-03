"""
API routes for the LLM Learning Platform.
Supports dual backends: 'custom' (NumPy) and 'pytorch' (GPU-accelerated).
"""

import os
import uuid
import pickle
import numpy as np
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.models.gpt import MicroGPT, GPTConfig
from app.core.trainer import TrainingEngine, TrainingConfig, DataLoader

# PyTorch imports (optional — gracefully degrade to CPU-only custom backend)
try:
    import torch
    from app.models.pytorch_gpt import PyTorchGPT, PyTorchGPTConfig, get_device, DEVICE
    from app.core.pytorch_trainer import PyTorchTrainingEngine, PyTorchTrainingConfig
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    DEVICE = None


# ============== Pydantic Models ==============

class GPTConfigRequest(BaseModel):
    vocab_size: int = Field(default=256, ge=16, le=10000)
    max_seq_len: int = Field(default=256, ge=32, le=2048)
    d_model: int = Field(default=128, ge=32, le=1024)
    num_layers: int = Field(default=4, ge=1, le=24)
    num_heads: int = Field(default=4, ge=1, le=32)
    d_ff: Optional[int] = None
    dropout: float = Field(default=0.1, ge=0.0, le=0.5)
    attention_dropout: float = Field(default=0.1, ge=0.0, le=0.5)
    activation: str = Field(default='gelu', pattern='^(gelu|relu|swiglu)$')
    norm_type: str = Field(default='rmsnorm', pattern='^(rmsnorm|layernorm)$')
    tie_weights: bool = True
    backend: str = Field(default='pytorch', pattern='^(custom|pytorch)$')


class TrainingConfigRequest(BaseModel):
    model_id: str
    batch_size: int = Field(default=32, ge=1, le=256)
    learning_rate: float = Field(default=3e-4, ge=1e-6, le=1e-1)
    min_learning_rate: float = Field(default=3e-5, ge=1e-7, le=1e-2)
    warmup_steps: int = Field(default=100, ge=0, le=10000)
    max_steps: int = Field(default=10000, ge=100, le=1000000)
    grad_clip: float = Field(default=1.0, ge=0.0, le=10.0)
    weight_decay: float = Field(default=0.1, ge=0.0, le=1.0)
    seq_length: int = Field(default=256, ge=32, le=1024)
    eval_interval: int = Field(default=100, ge=10, le=10000)
    checkpoint_interval: int = Field(default=1000, ge=100, le=50000)


class GenerationRequest(BaseModel):
    model_id: str
    prompt: str
    max_new_tokens: int = Field(default=100, ge=1, le=1000)
    temperature: float = Field(default=0.8, ge=0.1, le=2.0)
    top_k: Optional[int] = Field(default=40, ge=1, le=500)
    top_p: Optional[float] = Field(default=0.9, ge=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.0, ge=1.0, le=2.0)


class TokenizeRequest(BaseModel):
    text: str
    strategy: str = Field(default='character', pattern='^(character|word|bpe)$')


class ModelInfo(BaseModel):
    model_id: str
    config: Dict[str, Any]
    num_parameters: int
    created_at: str
    status: str


class TrainingStatus(BaseModel):
    session_id: str
    is_training: bool
    current_step: int
    current_epoch: int
    best_loss: float
    progress: float


# ============== In-Memory Storage ==============
# In production, this would be Redis or a database

_models: Dict[str, Any] = {}  # MicroGPT or PyTorchGPT
_model_configs: Dict[str, Any] = {}  # GPTConfig or PyTorchGPTConfig
_model_backends: Dict[str, str] = {}  # 'custom' or 'pytorch'
_training_engines: Dict[str, Any] = {}  # TrainingEngine or PyTorchTrainingEngine
_training_sessions: Dict[str, Dict[str, Any]] = {}

# Simple character-level tokenizer for demo
def simple_tokenize(text: str) -> List[int]:
    """Simple character-level tokenization."""
    return [ord(c) % 256 for c in text]


def simple_detokenize(tokens: List[int]) -> str:
    """Simple character-level detokenization."""
    return ''.join(chr(t) for t in tokens)


# ============== Router ==============

router = APIRouter(prefix="/api", tags=["api"])


# ============== Model Endpoints ==============

@router.post("/model/create", response_model=ModelInfo)
async def create_model(config: GPTConfigRequest):
    """Create a new GPT model. Supports 'custom' (NumPy) or 'pytorch' (GPU) backend."""
    model_id = str(uuid.uuid4())[:8]
    
    # Set default d_ff if not provided
    if config.d_ff is None:
        config.d_ff = config.d_model * 4
    
    backend = config.backend
    
    # Fall back to custom if PyTorch not available
    if backend == 'pytorch' and not PYTORCH_AVAILABLE:
        backend = 'custom'
    
    if backend == 'pytorch' and PYTORCH_AVAILABLE:
        # Create PyTorch model (GPU-accelerated)
        pt_config = PyTorchGPTConfig(
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_len,
            d_model=config.d_model,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            dropout=config.dropout,
            attention_dropout=config.attention_dropout,
            activation=config.activation,
            norm_type=config.norm_type,
            tie_weights=config.tie_weights,
        )
        model = PyTorchGPT(pt_config).to(DEVICE)
        num_params = model.get_num_parameters()
        model_config_dict = {
            'vocab_size': pt_config.vocab_size,
            'max_seq_len': pt_config.max_seq_len,
            'd_model': pt_config.d_model,
            'num_layers': pt_config.num_layers,
            'num_heads': pt_config.num_heads,
            'd_ff': pt_config.d_ff,
            'dropout': pt_config.dropout,
            'activation': pt_config.activation,
            'norm_type': pt_config.norm_type,
            'backend': 'pytorch',
            'device': str(DEVICE),
        }
        _models[model_id] = model
        _model_configs[model_id] = pt_config
        _model_backends[model_id] = 'pytorch'
    else:
        # Create custom NumPy model
        gpt_config = GPTConfig(
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_len,
            d_model=config.d_model,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            dropout=config.dropout,
            attention_dropout=config.attention_dropout,
            activation=config.activation,
            norm_type=config.norm_type,
            tie_weights=config.tie_weights,
        )
        model = MicroGPT(gpt_config)
        num_params = model.count_parameters()
        model_config_dict = model.get_config()
        model_config_dict['backend'] = 'custom'
        model_config_dict['device'] = 'cpu'
        _models[model_id] = model
        _model_configs[model_id] = gpt_config
        _model_backends[model_id] = 'custom'
    
    return ModelInfo(
        model_id=model_id,
        config=model_config_dict,
        num_parameters=num_params,
        created_at=datetime.now().isoformat(),
        status="created"
    )


@router.get("/model/{model_id}", response_model=ModelInfo)
async def get_model(model_id: str):
    """Get model information."""
    if model_id not in _models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model = _models[model_id]
    
    return ModelInfo(
        model_id=model_id,
        config=model.get_config(),
        num_parameters=model.count_parameters(),
        created_at=datetime.now().isoformat(),
        status="ready"
    )


@router.post("/model/{model_id}/reset")
async def reset_model(model_id: str):
    """Reset model parameters."""
    if model_id not in _models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    config = _model_configs[model_id]
    _models[model_id] = MicroGPT(config)
    
    return {"status": "reset", "model_id": model_id}


@router.delete("/model/{model_id}")
async def delete_model(model_id: str):
    """Delete a model."""
    if model_id in _models:
        del _models[model_id]
    if model_id in _model_configs:
        del _model_configs[model_id]
    if model_id in _training_engines:
        del _training_engines[model_id]
    
    return {"status": "deleted", "model_id": model_id}


@router.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List all models."""
    models = []
    for model_id, model in _models.items():
        models.append(ModelInfo(
            model_id=model_id,
            config=model.get_config(),
            num_parameters=model.count_parameters(),
            created_at=datetime.now().isoformat(),
            status="ready"
        ))
    return models


# ============== Training Endpoints ==============

@router.post("/training/start")
async def start_training(config: TrainingConfigRequest):
    """Start a training session. Uses PyTorch (GPU) or custom (CPU) engine based on model backend."""
    model_id = config.model_id
    
    if model_id not in _models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Create session ID
    session_id = str(uuid.uuid4())[:8]
    backend = _model_backends.get(model_id, 'custom')
    
    if backend == 'pytorch' and PYTORCH_AVAILABLE:
        # PyTorch GPU training engine
        train_config = PyTorchTrainingConfig(
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            min_learning_rate=config.min_learning_rate,
            warmup_steps=config.warmup_steps,
            max_steps=config.max_steps,
            grad_clip=config.grad_clip,
            weight_decay=config.weight_decay,
            seq_length=config.seq_length,
            eval_interval=config.eval_interval,
            checkpoint_interval=config.checkpoint_interval,
        )
        engine = PyTorchTrainingEngine(_models[model_id], train_config)
    else:
        # Custom NumPy training engine
        train_config = TrainingConfig(
            model_config=_model_configs[model_id],
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            min_learning_rate=config.min_learning_rate,
            warmup_steps=config.warmup_steps,
            max_steps=config.max_steps,
            grad_clip=config.grad_clip,
            weight_decay=config.weight_decay,
            seq_length=config.seq_length,
            eval_interval=config.eval_interval,
            checkpoint_interval=config.checkpoint_interval,
        )
        engine = TrainingEngine(_models[model_id], train_config)
    
    _training_engines[session_id] = engine
    
    # Store session info
    _training_sessions[session_id] = {
        'model_id': model_id,
        'config': config.dict(),
        'backend': backend,
        'started_at': datetime.now().isoformat(),
    }
    
    return {
        "session_id": session_id,
        "status": "started",
        "model_id": model_id,
        "backend": backend,
        "device": str(DEVICE) if backend == 'pytorch' else 'cpu',
    }


@router.post("/training/{session_id}/stop")
async def stop_training(session_id: str):
    """Stop a training session."""
    if session_id not in _training_engines:
        raise HTTPException(status_code=404, detail="Training session not found")
    
    engine = _training_engines[session_id]
    engine.stop()
    
    return {"status": "stopped", "session_id": session_id}


@router.get("/training/{session_id}/status", response_model=TrainingStatus)
async def get_training_status(session_id: str):
    """Get training session status."""
    if session_id not in _training_engines:
        raise HTTPException(status_code=404, detail="Training session not found")
    
    engine = _training_engines[session_id]
    status = engine.get_status()
    
    return TrainingStatus(
        session_id=session_id,
        is_training=status['is_training'],
        current_step=status['current_step'],
        current_epoch=status['current_epoch'],
        best_loss=status['best_loss'],
        progress=status['progress']
    )


@router.get("/training/{session_id}/history")
async def get_training_history(session_id: str, last_n: Optional[int] = None):
    """Get training metrics history."""
    if session_id not in _training_engines:
        raise HTTPException(status_code=404, detail="Training session not found")
    
    engine = _training_engines[session_id]
    history = engine.metrics_history
    
    if last_n:
        history = history[-last_n:]
    
    return {
        "session_id": session_id,
        "metrics": [m.to_dict() for m in history]
    }


# ============== Inference Endpoints ==============

@router.post("/inference/generate")
async def generate_text(request: GenerationRequest):
    """Generate text using a trained model. Supports both custom and PyTorch backends."""
    model_id = request.model_id
    
    if model_id not in _models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model = _models[model_id]
    backend = _model_backends.get(model_id, 'custom')
    
    # Tokenize input
    token_list = simple_tokenize(request.prompt)
    
    if backend == 'pytorch' and PYTORCH_AVAILABLE:
        # PyTorch generation (GPU)
        input_ids = torch.tensor([token_list], dtype=torch.long, device=DEVICE)
        output_ids = model.generate(
            input_ids,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
        )
        generated_text = simple_detokenize(output_ids[0].cpu().tolist())
    else:
        # Custom NumPy generation
        input_ids = np.array([token_list], dtype=np.int32)
        output_ids = model.generate(
            input_ids,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty
        )
        generated_text = simple_detokenize(output_ids[0].tolist())
    
    new_text = generated_text[len(request.prompt):]
    
    return {
        "model_id": model_id,
        "prompt": request.prompt,
        "generated_text": new_text,
        "full_text": generated_text,
        "tokens_generated": len(generated_text) - len(request.prompt),
        "backend": backend,
    }


@router.post("/inference/tokenize")
async def tokenize_text(request: TokenizeRequest):
    """Tokenize text using different strategies."""
    text = request.text
    strategy = request.strategy
    
    if strategy == 'character':
        tokens = [c for c in text]
        token_ids = simple_tokenize(text)
        
        return {
            "strategy": strategy,
            "text": text,
            "tokens": tokens,
            "token_ids": token_ids,
            "num_tokens": len(tokens),
            "vocabulary": list(set(tokens))
        }
    
    elif strategy == 'word':
        words = text.split()
        vocab = list(set(words))
        word_to_id = {w: i for i, w in enumerate(vocab)}
        token_ids = [word_to_id[w] for w in words]
        
        return {
            "strategy": strategy,
            "text": text,
            "tokens": words,
            "token_ids": token_ids,
            "num_tokens": len(words),
            "vocabulary": vocab,
            "vocab_size": len(vocab)
        }
    
    else:  # bpe - real byte-pair encoding
        from app.api.compute_routes import SimpleBPE
        bpe = SimpleBPE(num_merges=50)
        bpe_result = bpe.fit_and_tokenize(text)
        
        return {
            "strategy": strategy,
            "text": text,
            "tokens": bpe_result["tokens"],
            "token_ids": bpe_result["token_ids"],
            "num_tokens": bpe_result["num_tokens"],
            "vocabulary": bpe_result["vocabulary"],
            "vocab_size": bpe_result["vocab_size"],
            "merge_history": bpe_result.get("merge_history", []),
            "compression_ratio": bpe_result.get("compression_ratio", 1.0)
        }


@router.post("/inference/forward")
async def forward_pass(model_id: str, input_text: str):
    """Get model forward pass outputs and attention weights."""
    if model_id not in _models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model = _models[model_id]
    
    # Tokenize
    input_ids = np.array([simple_tokenize(input_text)], dtype=np.int32)
    
    # Forward pass with attention
    logits, loss, attention_weights = model.forward(
        input_ids, 
        return_attention=True
    )
    
    # Get top predictions
    probs = np.exp(logits.data[0, -1]) / np.sum(np.exp(logits.data[0, -1]))
    top_k_indices = np.argsort(probs)[-5:][::-1]
    top_k_probs = probs[top_k_indices]
    
    return {
        "model_id": model_id,
        "input": input_text,
        "logits_shape": logits.shape,
        "top_predictions": [
            {"token_id": int(idx), "probability": float(prob)}
            for idx, prob in zip(top_k_indices, top_k_probs)
        ],
        "attention_weights": [
            w.tolist() for w in attention_weights
        ] if attention_weights else None
    }


# ============== Visualization Endpoints ==============

@router.get("/viz/attention/{model_id}")
async def get_attention_data(
    model_id: str,
    text: str,
    layer: int = 0,
    head: int = 0
):
    """Get attention weights for visualization."""
    if model_id not in _models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model = _models[model_id]
    
    # Tokenize
    tokens = [c for c in text]
    input_ids = np.array([simple_tokenize(text)], dtype=np.int32)
    
    # Forward pass with attention
    _, _, attention_weights = model.forward(input_ids, return_attention=True)
    
    if not attention_weights or layer >= len(attention_weights):
        raise HTTPException(status_code=400, detail="Invalid layer index")
    
    # Get specific layer and head
    attn = attention_weights[layer][0, head]  # (seq_len, seq_len)
    
    return {
        "model_id": model_id,
        "text": text,
        "tokens": tokens,
        "layer": layer,
        "head": head,
        "attention_matrix": attn.tolist(),
        "shape": attn.shape
    }


@router.get("/viz/embeddings/{model_id}")
async def get_embeddings(model_id: str, method: str = 'pca'):
    """Get embedding visualization data."""
    if model_id not in _models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model = _models[model_id]
    embeddings = model.token_emb.weight.data  # (vocab_size, d_model)
    
    # Simple 2D projection (in production, use actual PCA/t-SNE)
    if method == 'pca':
        # Simple projection to 2D (first two principal components approx)
        if embeddings.shape[1] >= 2:
            projected = embeddings[:, :2]
        else:
            projected = np.pad(embeddings, ((0, 0), (0, 2 - embeddings.shape[1])))
    else:
        # Random projection for demo
        projected = np.random.randn(embeddings.shape[0], 2) * 0.5
    
    return {
        "model_id": model_id,
        "method": method,
        "vocab_size": embeddings.shape[0],
        "embedding_dim": embeddings.shape[1],
        "projections": projected.tolist(),
        "tokens": [chr(i) if i < 128 else f"<token_{i}>" for i in range(embeddings.shape[0])]
    }


# ============== GPU Status Endpoint ==============

@router.get("/gpu/status")
async def gpu_status():
    """Get GPU hardware and memory status."""
    if not PYTORCH_AVAILABLE:
        return {
            "pytorch_available": False,
            "cuda_available": False,
            "device": "cpu",
            "message": "PyTorch is not installed. Using custom NumPy backend.",
        }

    cuda_available = torch.cuda.is_available()
    result = {
        "pytorch_available": True,
        "pytorch_version": torch.__version__,
        "cuda_available": cuda_available,
        "device": str(DEVICE),
    }

    if cuda_available:
        props = torch.cuda.get_device_properties(0)
        result.update({
            "gpu_name": props.name,
            "gpu_total_memory_mb": round(props.total_memory / (1024 * 1024)),
            "gpu_allocated_memory_mb": round(torch.cuda.memory_allocated(0) / (1024 * 1024), 1),
            "gpu_reserved_memory_mb": round(torch.cuda.memory_reserved(0) / (1024 * 1024), 1),
            "gpu_utilization_percent": None,  # Would need pynvml for this
            "cuda_version": torch.version.cuda,
            "cudnn_version": str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else None,
        })
    else:
        result["message"] = "CUDA not available. PyTorch will use CPU."

    return result


# ============== WebSocket for Real-time Training ==============

class TrainingConnectionManager:
    """Manages WebSocket connections for training updates."""
    
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        if session_id not in self.active_connections:
            self.active_connections[session_id] = []
        self.active_connections[session_id].append(websocket)
    
    def disconnect(self, websocket: WebSocket, session_id: str):
        if session_id in self.active_connections:
            self.active_connections[session_id].remove(websocket)
    
    async def broadcast(self, session_id: str, message: dict):
        if session_id in self.active_connections:
            disconnected = []
            for connection in self.active_connections[session_id]:
                try:
                    await connection.send_json(message)
                except:
                    disconnected.append(connection)
            
            # Clean up disconnected
            for conn in disconnected:
                self.active_connections[session_id].remove(conn)


manager = TrainingConnectionManager()


@router.websocket("/ws/training/{session_id}")
async def training_websocket(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time training updates."""
    await manager.connect(websocket, session_id)
    
    try:
        while True:
            # Receive commands from client
            data = await websocket.receive_json()
            
            if data.get('command') == 'get_status':
                if session_id in _training_engines:
                    engine = _training_engines[session_id]
                    await websocket.send_json({
                        'type': 'status',
                        'data': engine.get_status()
                    })
            
            elif data.get('command') == 'stop':
                if session_id in _training_engines:
                    _training_engines[session_id].stop()
                    await websocket.send_json({
                        'type': 'stopped',
                        'message': 'Training stopped'
                    })
    
    except WebSocketDisconnect:
        manager.disconnect(websocket, session_id)


async def broadcast_metrics(session_id: str, metrics: dict):
    """Broadcast metrics to all connected clients."""
    await manager.broadcast(session_id, {
        'type': 'metrics',
        'data': metrics
    })


# ============== Datasets Endpoints ==============

BUILTIN_DATASETS = {
    "shakespeare": {
        "name": "Shakespeare",
        "description": "Complete works of William Shakespeare (~5.4MB)",
        "text": "To be, or not to be, that is the question...",
        "size": 5582212,
        "tokens": 1366766
    },
    "tiny_shakespeare": {
        "name": "Tiny Shakespeare", 
        "description": "First 1MB of Shakespeare (good for quick experiments)",
        "text": "To be, or not to be, that is the question...",
        "size": 1048576,
        "tokens": 256000
    },
    "war_and_peace": {
        "name": "War and Peace",
        "description": "Leo Tolstoy's War and Peace (~3.2MB)",
        "text": "Well, Prince, so Genoa and Lucca are now just family estates...",
        "size": 3355443,
        "tokens": 838860
    }
}

@router.get("/datasets")
async def list_datasets():
    """List all available datasets."""
    datasets = []
    for key, ds in BUILTIN_DATASETS.items():
        datasets.append({
            "id": key,
            "name": ds["name"],
            "description": ds["description"],
            "size": ds["size"],
            "tokens": ds["tokens"]
        })
    return {"datasets": datasets}


@router.get("/datasets/{dataset_id}")
async def get_dataset(dataset_id: str):
    """Get a specific dataset's information."""
    if dataset_id not in BUILTIN_DATASETS:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    ds = BUILTIN_DATASETS[dataset_id]
    return {
        "id": dataset_id,
        "name": ds["name"],
        "description": ds["description"],
        "size": ds["size"],
        "tokens": ds["tokens"],
        "preview": ds["text"][:200] + "..."
    }
