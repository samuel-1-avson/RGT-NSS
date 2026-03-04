"""
RLHF & Alignment API Router
"""

from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.rlhf import (
    RewardModel, PPOTrainer, DPOTrainer,
    RLHFConfig, PreferencePair,
)

router = APIRouter()

_reward_model = RewardModel()


class PreferencePairInput(BaseModel):
    prompt: str = "Write a helpful response"
    chosen: str = "Here is a helpful answer..."
    rejected: str = "I don't know"


class RewardTrainRequest(BaseModel):
    pairs: List[PreferencePairInput]
    learning_rate: float = Field(1e-3, ge=1e-6, le=1.0)
    epochs: int = Field(5, ge=1, le=50)


class PPOTrainRequest(BaseModel):
    num_steps: int = Field(20, ge=5, le=100)
    num_responses: int = Field(4, ge=2, le=16)
    epsilon: float = Field(0.2, ge=0.05, le=0.5)
    kl_coef: float = Field(0.1, ge=0.0, le=1.0)


class DPOTrainRequest(BaseModel):
    num_steps: int = Field(20, ge=5, le=100)
    beta: float = Field(0.1, ge=0.01, le=1.0)


class ScoreRequest(BaseModel):
    text: str


@router.post("/reward/score")
async def score_text(req: ScoreRequest):
    """Score a text sequence using the reward model."""
    token_ids = [ord(c) % _reward_model.vocab_size for c in req.text]
    score = _reward_model.score(token_ids)
    return {"text": req.text, "score": round(score, 4), "token_count": len(token_ids)}


@router.post("/reward/compare")
async def compare_responses(req: PreferencePairInput):
    """Compare chosen vs rejected response scores."""
    chosen_ids = [ord(c) % _reward_model.vocab_size for c in req.chosen]
    rejected_ids = [ord(c) % _reward_model.vocab_size for c in req.rejected]
    return {
        "chosen_score": round(_reward_model.score(chosen_ids), 4),
        "rejected_score": round(_reward_model.score(rejected_ids), 4),
        "preference_correct": _reward_model.score(chosen_ids) > _reward_model.score(rejected_ids),
    }


@router.post("/reward/train")
async def train_reward_model(req: RewardTrainRequest):
    """Train the reward model on preference pairs."""
    pairs = [PreferencePair(prompt=p.prompt, chosen=p.chosen, rejected=p.rejected) for p in req.pairs]
    history = _reward_model.train_on_preferences(pairs, lr=req.learning_rate, epochs=req.epochs)
    return {"training_history": history}


@router.post("/ppo/train")
async def train_ppo(req: PPOTrainRequest):
    """Run PPO training with real reward model scoring."""
    config = RLHFConfig(epsilon=req.epsilon, kl_coef=req.kl_coef)
    trainer = PPOTrainer(config)
    results = trainer.train(num_steps=req.num_steps, num_responses=req.num_responses)
    return {
        "config": {"epsilon": req.epsilon, "kl_coef": req.kl_coef},
        "steps": [
            {
                "step": r.step,
                "policy_loss": round(r.policy_loss, 4),
                "value_loss": round(r.value_loss, 4),
                "entropy": round(r.entropy, 4),
                "kl_div": round(r.kl_div, 4),
                "mean_reward": round(r.mean_reward, 4),
            }
            for r in results
        ],
    }


@router.post("/dpo/train")
async def train_dpo(req: DPOTrainRequest):
    """Run DPO training with real preference pair scoring."""
    trainer = DPOTrainer(beta=req.beta)
    results = trainer.train(num_steps=req.num_steps)
    return {"config": {"beta": req.beta}, "steps": results}


@router.get("/methods")
async def list_alignment_methods():
    """List available alignment methods with descriptions."""
    return {
        "methods": [
            {
                "name": "PPO",
                "full_name": "Proximal Policy Optimization",
                "description": "RL-based fine-tuning using a reward model and clipped policy gradient.",
                "components": ["SFT Model", "Reward Model", "PPO Optimizer"],
                "pros": ["Powerful and flexible", "Industry standard"],
                "cons": ["Complex pipeline", "Requires reward model", "Training instability"],
            },
            {
                "name": "DPO",
                "full_name": "Direct Preference Optimization",
                "description": "Directly optimize policy from preferences without explicit reward model.",
                "components": ["SFT Model", "Reference Model", "Preference Data"],
                "pros": ["Simpler pipeline", "No reward model needed", "More stable"],
                "cons": ["Less flexible", "Requires good reference model"],
            },
            {
                "name": "Constitutional AI",
                "full_name": "Constitutional AI",
                "description": "Self-improvement through AI-written critiques based on principles.",
                "components": ["Base Model", "Constitutional Principles", "Critique-Revision Loop"],
                "pros": ["Scalable", "Principle-based", "Less human labeling"],
                "cons": ["Requires good principles", "May be too conservative"],
            },
        ]
    }
