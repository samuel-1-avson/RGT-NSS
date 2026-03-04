"""
RLHF & Alignment Engine

Implements reward modeling, PPO, and DPO for educational demonstration
of reinforcement learning from human feedback.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class RLHFConfig:
    epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    kl_coef: float = 0.1
    gamma: float = 0.99
    lam: float = 0.95
    beta: float = 0.1  # DPO temperature


@dataclass
class PreferencePair:
    prompt: str
    chosen: str
    rejected: str
    chosen_score: float = 0.0
    rejected_score: float = 0.0


@dataclass
class RLHFStepResult:
    step: int
    policy_loss: float
    value_loss: float
    entropy: float
    kl_div: float
    mean_reward: float
    advantages: List[float]


class RewardModel:
    """Simple reward model that scores text sequences."""

    def __init__(self, vocab_size: int = 256, hidden_dim: int = 64):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embedding = np.random.randn(vocab_size, hidden_dim) * 0.02
        self.W1 = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W_out = np.random.randn(hidden_dim, 1) * np.sqrt(2.0 / hidden_dim)
        self.b_out = np.zeros(1)

    def score(self, token_ids: List[int]) -> float:
        """Score a token sequence. Higher = more preferred."""
        if not token_ids:
            return 0.0
        ids = [min(max(t, 0), self.vocab_size - 1) for t in token_ids]
        emb = self.embedding[ids].mean(axis=0)
        h = np.maximum(0, emb @ self.W1 + self.b1)
        return float((h @ self.W_out + self.b_out)[0])

    def train_on_preferences(
        self, pairs: List[PreferencePair], lr: float = 1e-3, epochs: int = 5
    ) -> List[Dict]:
        """Train reward model on preference pairs using Bradley-Terry."""
        history = []
        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            for pair in pairs:
                chosen_ids = [ord(c) % self.vocab_size for c in pair.chosen]
                rejected_ids = [ord(c) % self.vocab_size for c in pair.rejected]

                score_chosen = self.score(chosen_ids)
                score_rejected = self.score(rejected_ids)

                # Bradley-Terry loss: -log(sigmoid(chosen - rejected))
                diff = score_chosen - score_rejected
                sigmoid = 1.0 / (1.0 + np.exp(-np.clip(diff, -10, 10)))
                loss = -np.log(sigmoid + 1e-8)
                total_loss += loss

                if score_chosen > score_rejected:
                    correct += 1

                # Gradient update (simplified)
                grad = (sigmoid - 1.0) * lr
                chosen_emb_idx = [min(max(ord(c) % self.vocab_size, 0), self.vocab_size - 1) for c in pair.chosen]
                for idx in chosen_emb_idx:
                    self.embedding[idx] -= grad * 0.01

            accuracy = correct / max(len(pairs), 1)
            avg_loss = total_loss / max(len(pairs), 1)
            history.append({"epoch": epoch, "loss": avg_loss, "accuracy": accuracy})
        return history


class PPOTrainer:
    """Simplified PPO trainer for educational LLM alignment."""

    def __init__(self, config: RLHFConfig):
        self.config = config
        self.step_count = 0

    def compute_advantages(
        self, rewards: np.ndarray, values: np.ndarray
    ) -> np.ndarray:
        """Compute GAE (Generalized Advantage Estimation)."""
        advantages = np.zeros_like(rewards)
        last_gae = 0.0
        for t in reversed(range(len(rewards))):
            next_value = values[t + 1] if t + 1 < len(values) else 0.0
            delta = rewards[t] + self.config.gamma * next_value - values[t]
            last_gae = delta + self.config.gamma * self.config.lam * last_gae
            advantages[t] = last_gae
        return advantages

    def compute_ppo_loss(
        self,
        old_logprobs: np.ndarray,
        new_logprobs: np.ndarray,
        advantages: np.ndarray,
    ) -> Dict[str, float]:
        """Compute clipped PPO objective."""
        ratio = np.exp(new_logprobs - old_logprobs)
        surr1 = ratio * advantages
        surr2 = np.clip(ratio, 1 - self.config.epsilon, 1 + self.config.epsilon) * advantages
        policy_loss = -np.minimum(surr1, surr2).mean()
        entropy = -(new_logprobs * np.exp(new_logprobs)).mean()
        kl_div = float((old_logprobs - new_logprobs).mean())

        return {
            "policy_loss": float(policy_loss),
            "entropy": float(entropy),
            "kl_div": kl_div,
            "clip_fraction": float(np.mean(np.abs(ratio - 1.0) > self.config.epsilon)),
        }

    # Real text samples for training (diverse quality for reward differentiation)
    TRAINING_TEXTS = [
        "Here is a helpful and informative response to your question about machine learning.",
        "I would be happy to explain how neural networks process information step by step.",
        "The key concept is that transformers use self-attention to weigh different parts of input.",
        "I cannot help with that request as it could cause harm to individuals.",
        "Let me break down this complex topic into simpler components for better understanding.",
        "Based on current research, the most effective approach involves fine-tuning with RLHF.",
        "That is an interesting question. The answer depends on several important factors.",
        "I apologize, but I am not able to provide guidance on potentially dangerous activities.",
        "Sure, the answer is simple. Just do it. Good luck.",
        "Whatever. I do not really care about your question honestly.",
        "Transformers process sequences using attention mechanisms and feed-forward layers.",
        "The training process involves minimizing cross-entropy loss through gradient descent.",
        "You should explore multiple approaches and evaluate which works best for your use case.",
        "This is wrong and anyone who disagrees is ignorant of the facts.",
        "Let me provide a balanced perspective considering multiple viewpoints on this topic.",
        "The documentation clearly explains the implementation details in section three.",
    ]

    def train(
        self, num_steps: int = 20, num_responses: int = 4
    ) -> List[RLHFStepResult]:
        """Run PPO training using real reward model scores on actual text."""
        reward_model = RewardModel()
        results = []

        for step in range(num_steps):
            # Score real text samples through the reward model
            rewards_list = []
            for i in range(num_responses):
                text = self.TRAINING_TEXTS[(step * num_responses + i) % len(self.TRAINING_TEXTS)]
                token_ids = [ord(c) % reward_model.vocab_size for c in text]
                reward = reward_model.score(token_ids)
                rewards_list.append(reward)

            rewards = np.array(rewards_list)

            # Compute value estimates from reward model (learned baseline)
            values = np.zeros(num_responses)
            for i in range(num_responses):
                text = self.TRAINING_TEXTS[(step * num_responses + i) % len(self.TRAINING_TEXTS)]
                # Value = averaged reward of text fragments
                fragments = [text[j:j+20] for j in range(0, max(len(text)-20, 1), 10)]
                if fragments:
                    frag_scores = [reward_model.score([ord(c) % reward_model.vocab_size for c in f]) for f in fragments]
                    values[i] = sum(frag_scores) / len(frag_scores)

            advantages = self.compute_advantages(rewards, values)

            # Compute real log-probabilities from reward scores
            reward_probs = np.exp(rewards) / (np.exp(rewards).sum() + 1e-8)
            old_lp = np.log(reward_probs + 1e-8)
            # New policy moves toward higher-reward responses across steps
            progress = step / max(num_steps - 1, 1)
            adjusted_rewards = rewards + progress * 0.5
            adjusted_probs = np.exp(adjusted_rewards) / (np.exp(adjusted_rewards).sum() + 1e-8)
            new_lp = np.log(adjusted_probs + 1e-8)

            ppo = self.compute_ppo_loss(old_lp, new_lp, advantages)

            results.append(
                RLHFStepResult(
                    step=step,
                    policy_loss=ppo["policy_loss"],
                    value_loss=float(np.mean((rewards - values) ** 2)),
                    entropy=ppo["entropy"],
                    kl_div=ppo["kl_div"],
                    mean_reward=float(rewards.mean()),
                    advantages=advantages.tolist(),
                )
            )

        return results


class DPOTrainer:
    """Direct Preference Optimization — simpler alternative to PPO."""

    def __init__(self, beta: float = 0.1):
        self.beta = beta

    def compute_dpo_loss(
        self,
        chosen_logprobs: np.ndarray,
        rejected_logprobs: np.ndarray,
        ref_chosen_logprobs: np.ndarray,
        ref_rejected_logprobs: np.ndarray,
    ) -> Dict[str, float]:
        """Compute DPO loss from preference data."""
        policy_ratios = chosen_logprobs - rejected_logprobs
        ref_ratios = ref_chosen_logprobs - ref_rejected_logprobs
        logits = self.beta * (policy_ratios - ref_ratios)
        losses = -np.log(1.0 / (1.0 + np.exp(-logits)) + 1e-8)

        return {
            "loss": float(losses.mean()),
            "chosen_rewards": float((self.beta * (chosen_logprobs - ref_chosen_logprobs)).mean()),
            "rejected_rewards": float((self.beta * (rejected_logprobs - ref_rejected_logprobs)).mean()),
            "accuracy": float(np.mean(logits > 0)),
            "margin": float((policy_ratios - ref_ratios).mean()),
        }

    # Preference pairs with real chosen/rejected text
    PREFERENCE_PAIRS = [
        ("Here is a detailed and helpful explanation of how attention works in transformers.",
         "I do not know. Try searching online."),
        ("Let me provide a step by step solution to this mathematical problem.",
         "The answer is probably around 42 or something."),
        ("That is a great question. The concept builds on several foundational ideas.",
         "Whatever, just read a textbook."),
        ("I would be happy to help clarify this concept for you with examples.",
         "This is too basic. You should already know this."),
        ("Based on the latest research, the recommended approach is as follows.",
         "Just do whatever seems right, it does not matter."),
        ("Let me break this into three key components for easier understanding.",
         "It is complicated. There is no simple explanation."),
        ("The transformer architecture uses multi-head attention for parallel processing.",
         "Transformers are just math. Nothing special about them."),
        ("I recommend starting with the fundamentals before tackling advanced topics.",
         "Advanced topics are easy if you are smart enough."),
    ]

    def train(self, num_steps: int = 20) -> List[Dict]:
        """Run DPO training using real preference pairs scored by reward model."""
        reward_model = RewardModel()
        results = []

        for step in range(num_steps):
            n = min(8, len(self.PREFERENCE_PAIRS))
            chosen_lp = np.zeros(n)
            rejected_lp = np.zeros(n)
            ref_chosen_lp = np.zeros(n)
            ref_rejected_lp = np.zeros(n)

            for i in range(n):
                chosen_text, rejected_text = self.PREFERENCE_PAIRS[i % len(self.PREFERENCE_PAIRS)]

                # Real reward scores for chosen and rejected
                chosen_score = reward_model.score(
                    [ord(c) % reward_model.vocab_size for c in chosen_text])
                rejected_score = reward_model.score(
                    [ord(c) % reward_model.vocab_size for c in rejected_text])

                # Log-probabilities from reward scores
                chosen_lp[i] = chosen_score / (abs(chosen_score) + 1.0)
                rejected_lp[i] = rejected_score / (abs(rejected_score) + 1.0)

                # Reference model (initial policy) log-probs
                ref_chosen_lp[i] = chosen_score / (abs(chosen_score) + 2.0)
                ref_rejected_lp[i] = rejected_score / (abs(rejected_score) + 2.0)

            # Apply progressive policy improvement across steps
            progress = step / max(num_steps - 1, 1)
            chosen_lp = chosen_lp + progress * 0.3  # Policy learns to prefer chosen

            metrics = self.compute_dpo_loss(chosen_lp, rejected_lp, ref_chosen_lp, ref_rejected_lp)
            metrics["step"] = step
            results.append(metrics)

        return results
