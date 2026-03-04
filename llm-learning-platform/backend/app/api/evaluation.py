"""
Model Evaluation & Benchmarking API Router
"""

from typing import Dict, List, Optional
from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.core.evaluation import (
    compute_perplexity, compute_bleu, compute_rouge,
    run_benchmark_suite, compare_models,
)

router = APIRouter()


class BleuRequest(BaseModel):
    reference: str = "The cat sat on the mat"
    hypothesis: str = "The cat is on the mat"
    max_n: int = Field(4, ge=1, le=4)


class RougeRequest(BaseModel):
    reference: str = "The quick brown fox jumps over the lazy dog"
    hypothesis: str = "The fast brown fox leaps over the lazy dog"


class PerplexityRequest(BaseModel):
    losses: List[float] = [2.5, 2.3, 2.1, 1.9, 1.8]


class BenchmarkRequest(BaseModel):
    model_name: str = "MicroGPT"


class CompareRequest(BaseModel):
    model_names: List[str] = ["MicroGPT-nano", "MicroGPT-micro", "MicroGPT-small"]


@router.post("/bleu")
async def evaluate_bleu(req: BleuRequest):
    """Compute BLEU score between reference and hypothesis."""
    return compute_bleu(req.reference, req.hypothesis, req.max_n)


@router.post("/rouge")
async def evaluate_rouge(req: RougeRequest):
    """Compute ROUGE-1, ROUGE-2, ROUGE-L scores."""
    return compute_rouge(req.reference, req.hypothesis)


@router.post("/perplexity")
async def evaluate_perplexity(req: PerplexityRequest):
    """Compute perplexity from loss values."""
    ppl = compute_perplexity(req.losses)
    return {
        "perplexity": round(ppl, 2),
        "avg_loss": round(sum(req.losses) / max(len(req.losses), 1), 4),
        "num_samples": len(req.losses),
    }


@router.post("/benchmark")
async def run_benchmarks(req: BenchmarkRequest):
    """Run full benchmark suite on a model."""
    return run_benchmark_suite(req.model_name)


@router.post("/compare")
async def compare_model_benchmarks(req: CompareRequest):
    """Compare benchmark results across models."""
    return compare_models(req.model_names)


@router.get("/metrics")
async def list_metrics():
    """List available evaluation metrics."""
    return {
        "metrics": [
            {"name": "BLEU", "category": "generation", "description": "N-gram precision with brevity penalty", "range": "0-1"},
            {"name": "ROUGE-1", "category": "summarization", "description": "Unigram overlap (precision, recall, F1)", "range": "0-1"},
            {"name": "ROUGE-2", "category": "summarization", "description": "Bigram overlap", "range": "0-1"},
            {"name": "ROUGE-L", "category": "summarization", "description": "Longest common subsequence", "range": "0-1"},
            {"name": "Perplexity", "category": "language_modeling", "description": "Exponentiated cross-entropy loss", "range": "1-∞ (lower is better)"},
            {"name": "HellaSwag", "category": "reasoning", "description": "Commonsense NLI / sentence completion", "range": "0-100%"},
            {"name": "MMLU", "category": "knowledge", "description": "Multi-task language understanding", "range": "0-100%"},
        ]
    }
