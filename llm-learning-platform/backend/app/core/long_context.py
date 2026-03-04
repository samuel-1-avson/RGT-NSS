"""
Long Context Techniques Engine

Implements RoPE scaling (YaRN, NTK), ALiBi, and context extension analysis.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class RoPEConfig:
    dim: int = 128
    base: float = 10000.0
    max_position: int = 2048
    scaling_factor: float = 1.0


class RoPEAnalyzer:
    """Analyze and visualize Rotary Position Embeddings."""

    def __init__(self, config: RoPEConfig):
        self.config = config

    def compute_frequencies(self, scaling: str = "none") -> Dict:
        """Compute RoPE frequencies with optional scaling."""
        dim = self.config.dim
        base = self.config.base

        inv_freq = 1.0 / (base ** (np.arange(0, dim, 2).astype(np.float64) / dim))

        if scaling == "linear":
            inv_freq = inv_freq / self.config.scaling_factor
        elif scaling == "ntk":
            # NTK-aware scaling
            base_scaled = base * (self.config.scaling_factor ** (dim / (dim - 2)))
            inv_freq = 1.0 / (base_scaled ** (np.arange(0, dim, 2).astype(np.float64) / dim))
        elif scaling == "yarn":
            # YaRN: blend of high and low frequencies
            beta_fast, beta_slow = 32, 1
            low_freq_factor = 1.0
            high_freq_factor = 4.0
            old_context = self.config.max_position
            new_context = int(old_context * self.config.scaling_factor)
            low_freq_wavelen = old_context / low_freq_factor
            high_freq_wavelen = old_context / high_freq_factor

            wavelens = 2 * np.pi / inv_freq
            yarn_inv_freq = np.copy(inv_freq)
            for i, wavelen in enumerate(wavelens):
                if wavelen < high_freq_wavelen:
                    pass  # Keep original
                elif wavelen > low_freq_wavelen:
                    yarn_inv_freq[i] = inv_freq[i] / self.config.scaling_factor
                else:
                    smooth = (old_context / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
                    yarn_inv_freq[i] = (1 - smooth) * inv_freq[i] / self.config.scaling_factor + smooth * inv_freq[i]
            inv_freq = yarn_inv_freq

        max_pos = int(self.config.max_position * max(self.config.scaling_factor, 1))
        positions = np.arange(max_pos)
        freqs = np.outer(positions, inv_freq)

        # Sample attention decay at different distances
        decay_distances = [1, 10, 50, 100, 500, 1000, 2048, 4096]
        decay_scores = []
        for d in decay_distances:
            if d < max_pos:
                cos_sim = np.cos(freqs[0] - freqs[min(d, max_pos - 1)]).mean()
                decay_scores.append({"distance": d, "similarity": round(float(cos_sim), 4)})

        return {
            "scaling": scaling,
            "dim": dim,
            "max_position": max_pos,
            "num_frequencies": len(inv_freq),
            "frequency_range": [float(inv_freq.min()), float(inv_freq.max())],
            "wavelength_range": [float((2 * np.pi / inv_freq).min()), float((2 * np.pi / inv_freq).max())],
            "positional_decay": decay_scores,
            "frequency_values": inv_freq[:8].tolist(),
        }

    def compare_scaling_methods(self) -> Dict:
        """Compare different RoPE scaling methods."""
        methods = {}
        for method in ["none", "linear", "ntk", "yarn"]:
            methods[method] = self.compute_frequencies(method)
        return {"base_config": {"dim": self.config.dim, "scaling_factor": self.config.scaling_factor}, "methods": methods}


class ALiBiAnalyzer:
    """Analyze Attention with Linear Biases."""

    def __init__(self, num_heads: int = 8, max_len: int = 2048):
        self.num_heads = num_heads
        self.max_len = max_len
        self.slopes = self._compute_slopes()

    def _compute_slopes(self) -> np.ndarray:
        """Compute head-specific slopes (geometric sequence)."""
        ratio = 2 ** (-8 / self.num_heads)
        return np.array([ratio ** (i + 1) for i in range(self.num_heads)])

    def compute_bias_matrix(self, seq_len: int = 16) -> Dict:
        """Compute ALiBi bias matrix for visualization."""
        seq_len = min(seq_len, 64)  # Cap for visualization
        positions = np.arange(seq_len)
        distances = np.abs(positions[:, None] - positions[None, :])

        head_biases = []
        for h in range(self.num_heads):
            bias = -self.slopes[h] * distances
            head_biases.append({
                "head": h,
                "slope": round(float(self.slopes[h]), 6),
                "bias_matrix": np.round(bias, 4).tolist(),
                "effective_window": int(1.0 / max(self.slopes[h], 1e-8)),
            })

        return {
            "num_heads": self.num_heads,
            "seq_len": seq_len,
            "slopes": self.slopes.tolist(),
            "heads": head_biases,
        }

    def analyze_context_extension(self, train_len: int = 2048, test_lengths: Optional[List[int]] = None) -> Dict:
        """Analyze ALiBi's ability to extrapolate beyond training length."""
        if test_lengths is None:
            test_lengths = [512, 1024, 2048, 4096, 8192, 16384]

        results = []
        for length in test_lengths:
            ratio = length / train_len
            # ALiBi naturally handles extrapolation
            estimated_quality = 1.0 if ratio <= 1 else max(0.5, 1.0 - 0.1 * np.log2(ratio))
            results.append({
                "context_length": length,
                "ratio_to_training": round(ratio, 2),
                "estimated_quality": round(float(estimated_quality), 3),
                "extrapolating": length > train_len,
            })

        return {"train_length": train_len, "method": "ALiBi", "results": results}


def compare_position_methods(seq_len: int = 2048, dim: int = 128, num_heads: int = 8) -> Dict:
    """Compare all positional encoding methods for context extension."""
    methods = {
        "sinusoidal": {
            "max_extrapolation": "Unlimited (but quality degrades)",
            "trainable_params": 0,
            "relative_position": False,
            "description": "Fixed sin/cos functions at different frequencies",
        },
        "learned": {
            "max_extrapolation": "None (fixed to training length)",
            "trainable_params": seq_len * dim,
            "relative_position": False,
            "description": "Learned embedding per position",
        },
        "rope": {
            "max_extrapolation": "Limited (quality degrades beyond 2x training)",
            "trainable_params": 0,
            "relative_position": True,
            "description": "Rotary embeddings encoding relative positions",
        },
        "rope_yarn": {
            "max_extrapolation": f"Up to {seq_len * 16} with YaRN scaling",
            "trainable_params": 0,
            "relative_position": True,
            "description": "RoPE with YaRN temperature scaling",
        },
        "alibi": {
            "max_extrapolation": "Excellent (linear bias naturally extrapolates)",
            "trainable_params": 0,
            "relative_position": True,
            "description": "Linear attention bias based on distance",
        },
    }
    return {"seq_len": seq_len, "dim": dim, "num_heads": num_heads, "methods": methods}
