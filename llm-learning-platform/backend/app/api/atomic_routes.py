"""
API routes for atomic GPT operations.
Provides step-by-step computation visualization.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from app.core.atomic_gpt import Value, AtomicGPT, AtomicGPTConfig, Trainer, Adam, CharacterDataset

router = APIRouter(prefix="/api/atomic", tags=["atomic"])


class StepRequest(BaseModel):
    operation: str
    inputs: List[float]


class StepResponse(BaseModel):
    operation: str
    inputs: List[float]
    output: float
    local_grads: List[float]
    computation_graph: Dict[str, Any]


class AtomicTrainRequest(BaseModel):
    text: str
    vocab_size: int = 27
    n_embd: int = 16
    n_layer: int = 1
    n_head: int = 4
    num_steps: int = 10


class AtomicTrainResponse(BaseModel):
    step: int
    loss: float
    param_stats: Dict[str, float]
    gradients: List[float]


class ForwardPassRequest(BaseModel):
    token_id: int
    pos_id: int
    include_intermediates: bool = True


class ForwardPassResponse(BaseModel):
    token_id: int
    pos_id: int
    logits: List[float]
    intermediates: Optional[Dict[str, Any]]


# Store atomic model instances
_atomic_models: Dict[str, AtomicGPT] = {}
_atomic_trainers: Dict[str, Trainer] = {}


@router.post("/compute/step", response_model=StepResponse)
async def compute_step(request: StepRequest):
    """
    Execute a single computation step and return gradient information.
    
    Useful for visualizing how operations build computation graphs.
    """
    # Create Value objects
    inputs = [Value(x) for x in request.inputs]
    
    # Perform operation
    if request.operation == 'add':
        result = inputs[0] + inputs[1]
    elif request.operation == 'mul':
        result = inputs[0] * inputs[1]
    elif request.operation == 'relu':
        result = inputs[0].relu()
    elif request.operation == 'exp':
        result = inputs[0].exp()
    elif request.operation == 'log':
        result = inputs[0].log()
    else:
        raise HTTPException(status_code=400, detail=f"Unknown operation: {request.operation}")
    
    # Build computation graph info
    graph = {
        'node_id': id(result),
        'operation': result._op,
        'children': [id(child) for child in result._children],
    }
    
    return StepResponse(
        operation=request.operation,
        inputs=request.inputs,
        output=result.data,
        local_grads=list(result._local_grads),
        computation_graph=graph
    )


@router.post("/model/create")
async def create_atomic_model(config: AtomicGPTConfig):
    """Create a new atomic GPT model for step-by-step inspection."""
    model_id = f"atomic_{len(_atomic_models)}"
    model = AtomicGPT(config)
    _atomic_models[model_id] = model
    
    return {
        "model_id": model_id,
        "num_parameters": model.count_parameters(),
        "config": {
            "vocab_size": config.vocab_size,
            "n_layer": config.n_layer,
            "n_embd": config.n_embd,
            "n_head": config.n_head,
            "block_size": config.block_size,
        }
    }


@router.post("/model/{model_id}/forward", response_model=ForwardPassResponse)
async def atomic_forward(model_id: str, request: ForwardPassRequest):
    """
    Execute forward pass with optional intermediate values.
    
    When include_intermediates=True, returns:
    - Q, K, V projections
    - Attention scores
    - Attention weights
    - MLP activations
    """
    if model_id not in _atomic_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model = _atomic_models[model_id]
    
    # Initialize KV cache
    keys = [[] for _ in range(model.config.n_layer)]
    values = [[] for _ in range(model.config.n_layer)]
    
    # Forward pass
    logits = model.forward(request.token_id, request.pos_id, keys, values)
    
    response = ForwardPassResponse(
        token_id=request.token_id,
        pos_id=request.pos_id,
        logits=[l.data for l in logits],
        intermediates=None
    )
    
    if request.include_intermediates:
        # Collect intermediate values
        # Note: This requires modifying the atomic GPT to store intermediates
        # For now, return structure only
        response.intermediates = {
            "note": "For full intermediates, use atomic_gpt.py directly",
            "num_layers": model.config.n_layer,
            "embedding_dim": model.config.n_embd,
            "num_heads": model.config.n_head,
        }
    
    return response


@router.post("/model/{model_id}/train_step", response_model=AtomicTrainResponse)
async def atomic_train_step(model_id: str, tokens: List[int]):
    """
    Execute single training step with full gradient visibility.
    
    Returns loss and gradient statistics for visualization.
    """
    if model_id not in _atomic_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    if model_id not in _atomic_trainers:
        model = _atomic_models[model_id]
        optimizer = Adam(model.params, lr=0.01)
        _atomic_trainers[model_id] = Trainer(model, optimizer)
    
    trainer = _atomic_trainers[model_id]
    loss = trainer.train_step(tokens)
    
    # Collect gradient statistics
    model = _atomic_models[model_id]
    grads = [p.grad for p in model.params]
    
    return AtomicTrainResponse(
        step=trainer.step,
        loss=loss,
        param_stats={
            "mean_grad": sum(grads) / len(grads),
            "max_grad": max(grads),
            "min_grad": min(grads),
            "num_params": len(model.params),
        },
        gradients=grads[:10]  # First 10 gradients only
    )


@router.post("/demo/gradient_flow")
async def demo_gradient_flow():
    """
    Demonstrate gradient flow through a simple computation.
    
    Perfect for visualizing backpropagation step-by-step.
    """
    # Simple computation: f(x, y) = (x * y + 2)^2
    x = Value(3.0, label='x')
    y = Value(4.0, label='y')
    
    z = x * y
    z._label = 'z = x * y'
    
    w = z + 2
    w._label = 'w = z + 2'
    
    f = w ** 2
    f._label = 'f = w^2'
    
    # Backward pass
    f.backward()
    
    # Build computation graph
    nodes = []
    for node in [x, y, z, w, f]:
        nodes.append({
            'id': id(node),
            'label': node._label,
            'data': node.data,
            'grad': node.grad,
            'op': node._op,
        })
    
    return {
        "computation": "f(x, y) = (x * y + 2)^2",
        "inputs": {"x": 3.0, "y": 4.0},
        "output": f.data,
        "gradients": {
            "df/dx": x.grad,
            "df/dy": y.grad,
        },
        "computation_graph": nodes,
        "explanation": {
            "forward": [
                "z = x * y = 3 * 4 = 12",
                "w = z + 2 = 12 + 2 = 14", 
                "f = w^2 = 14^2 = 196"
            ],
            "backward": [
                "df/dw = 2*w = 28",
                "df/dz = df/dw * dw/dz = 28 * 1 = 28",
                "df/dx = df/dz * dz/dx = 28 * y = 28 * 4 = 112",
                "df/dy = df/dz * dz/dy = 28 * x = 28 * 3 = 84"
            ]
        }
    }


@router.get("/educational/computation_types")
async def get_computation_types():
    """Get educational explanation of different computation types."""
    return {
        "operations": [
            {
                "name": "Addition",
                "formula": "f(x, y) = x + y",
                "local_grads": ["∂f/∂x = 1", "∂f/∂y = 1"],
                "intuition": "Sum distributes gradient equally to both inputs"
            },
            {
                "name": "Multiplication", 
                "formula": "f(x, y) = x * y",
                "local_grads": ["∂f/∂x = y", "∂f/∂y = x"],
                "intuition": "Gradient flows proportional to the other input"
            },
            {
                "name": "ReLU",
                "formula": "f(x) = max(0, x)",
                "local_grads": ["∂f/∂x = 1 if x > 0 else 0"],
                "intuition": "Gradient flows only through positive values"
            },
            {
                "name": "Power",
                "formula": "f(x) = x^n",
                "local_grads": ["∂f/∂x = n * x^(n-1)"],
                "intuition": "Gradient depends on current value raised to power"
            }
        ]
    }
