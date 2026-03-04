"""
Educational Modules API

Manage learning modules / labs and track user progress through them.
"""

from enum import Enum
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()


class ModuleDifficulty(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class ModuleCategory(str, Enum):
    FOUNDATIONS = "foundations"
    ARCHITECTURE = "architecture"
    TRAINING = "training"
    ADVANCED = "advanced"
    FRONTIER = "frontier"


_MODULES = [
    {
        "id": "tokenization-lab",
        "title": "Tokenization Lab",
        "description": "Interactive exploration of tokenization algorithms: BPE, WordPiece, SentencePiece.",
        "category": ModuleCategory.FOUNDATIONS,
        "difficulty": ModuleDifficulty.BEGINNER,
        "xp_reward": 100,
        "estimated_minutes": 45,
        "prerequisites": [],
        "topics": ["BPE", "WordPiece", "Unicode", "Subword Tokenization"],
        "order": 1,
    },
    {
        "id": "embedding-explorer",
        "title": "Embedding Explorer",
        "description": "Visualize and interact with word embeddings, positional encodings, and semantic space.",
        "category": ModuleCategory.FOUNDATIONS,
        "difficulty": ModuleDifficulty.BEGINNER,
        "xp_reward": 120,
        "estimated_minutes": 60,
        "prerequisites": ["tokenization-lab"],
        "topics": ["Embeddings", "Positional Encoding", "Cosine Similarity", "PCA"],
        "order": 2,
    },
    {
        "id": "attention-visualizer",
        "title": "Attention Visualizer",
        "description": "Step through attention computation with interactive heatmaps and animations.",
        "category": ModuleCategory.ARCHITECTURE,
        "difficulty": ModuleDifficulty.INTERMEDIATE,
        "xp_reward": 200,
        "estimated_minutes": 90,
        "prerequisites": ["embedding-explorer"],
        "topics": ["Self-Attention", "Multi-Head", "Causal Masking", "KV-Cache"],
        "order": 3,
    },
    {
        "id": "transformer-builder",
        "title": "Transformer Builder",
        "description": "Assemble transformer blocks interactively — choose norms, activations, and architectures.",
        "category": ModuleCategory.ARCHITECTURE,
        "difficulty": ModuleDifficulty.INTERMEDIATE,
        "xp_reward": 250,
        "estimated_minutes": 120,
        "prerequisites": ["attention-visualizer"],
        "topics": ["LayerNorm", "RMSNorm", "GELU", "SwiGLU", "Residual Connections"],
        "order": 4,
    },
    {
        "id": "training-dashboard",
        "title": "Training Dashboard",
        "description": "Train models end-to-end with real-time loss curves and hyperparameter tuning.",
        "category": ModuleCategory.TRAINING,
        "difficulty": ModuleDifficulty.INTERMEDIATE,
        "xp_reward": 300,
        "estimated_minutes": 120,
        "prerequisites": ["transformer-builder"],
        "topics": ["Backpropagation", "AdamW", "Learning Rate Schedule", "Gradient Clipping"],
        "order": 5,
    },
    {
        "id": "inference-playground",
        "title": "Inference Playground",
        "description": "Explore autoregressive generation with temperature, top-k, and top-p sampling.",
        "category": ModuleCategory.TRAINING,
        "difficulty": ModuleDifficulty.INTERMEDIATE,
        "xp_reward": 200,
        "estimated_minutes": 60,
        "prerequisites": ["training-dashboard"],
        "topics": ["Autoregressive Generation", "Temperature", "Top-k", "Top-p", "Beam Search"],
        "order": 6,
    },
    {
        "id": "rlhf-lab",
        "title": "RLHF & Alignment Lab",
        "description": "Explore reinforcement learning from human feedback, PPO, DPO, and alignment techniques.",
        "category": ModuleCategory.ADVANCED,
        "difficulty": ModuleDifficulty.ADVANCED,
        "xp_reward": 400,
        "estimated_minutes": 150,
        "prerequisites": ["training-dashboard"],
        "topics": ["RLHF", "PPO", "DPO", "Reward Modeling", "Constitutional AI"],
        "order": 7,
    },
    {
        "id": "lora-qlora-studio",
        "title": "LoRA & QLoRA Studio",
        "description": "Learn parameter-efficient fine-tuning with low-rank adaptation.",
        "category": ModuleCategory.ADVANCED,
        "difficulty": ModuleDifficulty.ADVANCED,
        "xp_reward": 350,
        "estimated_minutes": 120,
        "prerequisites": ["training-dashboard"],
        "topics": ["LoRA", "QLoRA", "Adapter Layers", "Rank Selection"],
        "order": 8,
    },
    {
        "id": "evaluation-center",
        "title": "Evaluation Center",
        "description": "Benchmark models with BLEU, ROUGE, perplexity, and custom metrics.",
        "category": ModuleCategory.ADVANCED,
        "difficulty": ModuleDifficulty.ADVANCED,
        "xp_reward": 300,
        "estimated_minutes": 90,
        "prerequisites": ["inference-playground"],
        "topics": ["BLEU", "ROUGE", "Perplexity", "Human Evaluation", "LLM-as-Judge"],
        "order": 9,
    },
    {
        "id": "prompt-engineering",
        "title": "Prompt Engineering Lab",
        "description": "Master prompting techniques: few-shot, chain-of-thought, structured output.",
        "category": ModuleCategory.ADVANCED,
        "difficulty": ModuleDifficulty.INTERMEDIATE,
        "xp_reward": 250,
        "estimated_minutes": 90,
        "prerequisites": ["inference-playground"],
        "topics": ["Few-shot", "Chain-of-Thought", "System Prompts", "Output Parsing"],
        "order": 10,
    },
    {
        "id": "mechanistic-interpretability",
        "title": "Mechanistic Interpretability",
        "description": "Probe attention heads, neurons, and circuits inside the model.",
        "category": ModuleCategory.FRONTIER,
        "difficulty": ModuleDifficulty.EXPERT,
        "xp_reward": 500,
        "estimated_minutes": 180,
        "prerequisites": ["attention-visualizer", "training-dashboard"],
        "topics": ["Probing", "Activation Patching", "Circuit Discovery", "Superposition"],
        "order": 11,
    },
    {
        "id": "quantization-lab",
        "title": "Quantization Lab",
        "description": "Explore INT8, INT4, and mixed-precision quantization effects on model quality.",
        "category": ModuleCategory.FRONTIER,
        "difficulty": ModuleDifficulty.EXPERT,
        "xp_reward": 400,
        "estimated_minutes": 120,
        "prerequisites": ["training-dashboard"],
        "topics": ["INT8", "INT4", "GPTQ", "AWQ", "Dynamic Quantization"],
        "order": 12,
    },
]


@router.get("")
async def list_modules(
    category: Optional[str] = None,
    difficulty: Optional[str] = None,
):
    """List all available educational modules with optional filtering."""
    modules = _MODULES
    if category:
        modules = [m for m in modules if m["category"] == category]
    if difficulty:
        modules = [m for m in modules if m["difficulty"] == difficulty]
    return {"modules": modules, "total": len(modules)}


@router.get("/{module_id}")
async def get_module(module_id: str):
    """Get detailed information about a specific module."""
    module = next((m for m in _MODULES if m["id"] == module_id), None)
    if not module:
        raise HTTPException(status_code=404, detail="Module not found")
    return module


@router.get("/{module_id}/prerequisites")
async def get_prerequisites(module_id: str):
    """Get prerequisite chain for a module."""
    module = next((m for m in _MODULES if m["id"] == module_id), None)
    if not module:
        raise HTTPException(status_code=404, detail="Module not found")

    prereqs = []
    to_check = list(module["prerequisites"])
    seen = set()
    while to_check:
        pid = to_check.pop(0)
        if pid in seen:
            continue
        seen.add(pid)
        p = next((m for m in _MODULES if m["id"] == pid), None)
        if p:
            prereqs.append({"id": pid, "title": p["title"]})
            to_check.extend(p["prerequisites"])

    return {"module_id": module_id, "prerequisites": prereqs}


@router.get("/categories/list")
async def list_categories():
    """List module categories with counts."""
    counts = {}
    for m in _MODULES:
        cat = m["category"]
        counts[cat] = counts.get(cat, 0) + 1
    return {"categories": counts}
