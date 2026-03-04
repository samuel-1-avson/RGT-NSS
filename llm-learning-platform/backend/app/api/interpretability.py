"""
Mechanistic Interpretability API Router
"""

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.core.interpretability import LogitLens, ActivationPatcher, NeuronAnalyzer, CircuitTracer

router = APIRouter()


class LogitLensRequest(BaseModel):
    text: str = "The capital of France is"
    num_layers: int = Field(6, ge=2, le=24)
    top_k: int = Field(5, ge=1, le=20)


class PatchRequest(BaseModel):
    clean_text: str = "The Eiffel Tower is in Paris"
    corrupted_text: str = "The Eiffel Tower is in London"
    target_position: int = -1


class NeuronRequest(BaseModel):
    text: str = "Hello world"
    layer: int = Field(0, ge=0, le=23)
    top_k: int = Field(10, ge=1, le=50)


class CircuitRequest(BaseModel):
    text: str = "The cat sat on the mat"
    target_position: int = -1
    num_layers: int = Field(6, ge=2, le=24)
    num_heads: int = Field(8, ge=1, le=32)


@router.post("/logit-lens")
async def run_logit_lens(req: LogitLensRequest):
    """Probe model predictions at each layer."""
    lens = LogitLens(num_layers=req.num_layers)
    return lens.probe_all_layers(req.text, top_k=req.top_k)


@router.post("/activation-patching")
async def run_activation_patching(req: PatchRequest):
    """Run activation patching to identify causal components."""
    patcher = ActivationPatcher(num_layers=6)
    return patcher.patch_and_measure(req.clean_text, req.corrupted_text, req.target_position)


@router.post("/neurons")
async def analyze_neurons(req: NeuronRequest):
    """Analyze neuron activations for given input."""
    analyzer = NeuronAnalyzer()
    return analyzer.analyze_neurons(req.text, layer=req.layer, top_k=req.top_k)


@router.post("/circuits")
async def trace_circuits(req: CircuitRequest):
    """Trace computation circuits through the model."""
    tracer = CircuitTracer(num_layers=req.num_layers, num_heads=req.num_heads)
    return tracer.trace_circuit(req.text, req.target_position)


@router.get("/tools")
async def list_interpretability_tools():
    """List available interpretability tools."""
    return {
        "tools": [
            {
                "name": "Logit Lens",
                "description": "Project intermediate layer representations to vocabulary space",
                "use_case": "See how predictions evolve layer by layer",
            },
            {
                "name": "Activation Patching",
                "description": "Replace activations from clean run with corrupted run to measure causal effect",
                "use_case": "Identify which components are responsible for specific behaviors",
            },
            {
                "name": "Neuron Analysis",
                "description": "Examine individual neuron activation patterns",
                "use_case": "Find specialized or dead neurons in MLP layers",
            },
            {
                "name": "Circuit Tracing",
                "description": "Trace information flow through attention heads and MLPs",
                "use_case": "Discover computational circuits and head specialization",
            },
        ]
    }
