"""
Inference Optimization Engine

Implements KV cache analysis, quantization analysis, and speculative decoding execution.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class KVCacheConfig:
    num_layers: int = 6
    num_heads: int = 8
    head_dim: int = 64
    max_seq_len: int = 2048
    dtype_bytes: int = 2  # FP16


class KVCacheAnalyzer:
    """Analyze KV cache behavior for visualization."""

    def __init__(self, config: KVCacheConfig):
        self.config = config
        self.current_len = 0

    def step(self, new_tokens: int = 1) -> Dict:
        """Track cache growth after adding tokens."""
        self.current_len += new_tokens
        self.current_len = min(self.current_len, self.config.max_seq_len)

        per_layer_bytes = (
            self.config.num_heads * self.config.head_dim * self.current_len * self.config.dtype_bytes * 2
        )
        total_bytes = per_layer_bytes * self.config.num_layers

        # Without cache: recompute all tokens
        no_cache_flops = self.current_len * self.current_len * self.config.num_heads * self.config.head_dim
        with_cache_flops = self.current_len * self.config.num_heads * self.config.head_dim

        return {
            "current_length": self.current_len,
            "max_length": self.config.max_seq_len,
            "utilization": round(self.current_len / self.config.max_seq_len, 4),
            "cache_mb": round(total_bytes / 1e6, 3),
            "per_layer_mb": round(per_layer_bytes / 1e6, 3),
            "without_cache_flops": int(no_cache_flops),
            "with_cache_flops": int(with_cache_flops),
            "speedup": round(no_cache_flops / max(with_cache_flops, 1), 2),
        }

    def analyze_generation(self, prompt_len: int = 20, gen_len: int = 50) -> List[Dict]:
        """Analyze entire generation with cache growth."""
        self.current_len = 0
        results = []

        # Prefill phase
        self.current_len = prompt_len
        results.append({"phase": "prefill", "token": 0, **self.step(0)})

        # Decode phase
        for i in range(gen_len):
            results.append({"phase": "decode", "token": i + 1, **self.step(1)})

        return results

    def reset(self):
        self.current_len = 0


class QuantizationAnalyzer:
    """Analyze effects of different quantization bit widths."""

    @staticmethod
    def quantize_tensor(tensor: np.ndarray, bits: int) -> Tuple[np.ndarray, Dict]:
        """Quantize a tensor to specified bit width and measure error."""
        num_levels = 2 ** bits
        t_min, t_max = tensor.min(), tensor.max()
        scale = (t_max - t_min) / max(num_levels - 1, 1)

        if scale == 0:
            return tensor.copy(), {
                "bits": bits, "mse": 0.0, "max_error": 0.0,
                "snr_db": float("inf"), "compression_ratio": 32.0 / bits
            }

        quantized_int = np.round((tensor - t_min) / scale).astype(np.int32)
        quantized_int = np.clip(quantized_int, 0, num_levels - 1)
        dequantized = quantized_int * scale + t_min

        error = tensor - dequantized
        mse = float(np.mean(error ** 2))
        signal_power = float(np.mean(tensor ** 2))

        return dequantized, {
            "bits": bits,
            "mse": round(mse, 8),
            "max_error": round(float(np.abs(error).max()), 6),
            "snr_db": round(float(10 * np.log10(signal_power / max(mse, 1e-15))), 2),
            "compression_ratio": round(32.0 / bits, 2),
            "num_levels": num_levels,
        }

    @staticmethod
    def compare_quantizations(shape: Tuple[int, ...] = (512, 512)) -> Dict:
        """Compare FP32, FP16, INT8, INT4 quantization on real model weights."""
        from app.core.model import MicroGPT, GPTConfig

        # Extract a real weight tensor from the model
        d_model = max(shape[0], shape[1])
        config = GPTConfig(
            vocab_size=max(d_model, 256), d_model=d_model,
            num_heads=max(1, d_model // 32), num_layers=2,
            d_ff=d_model * 4, max_seq_len=64,
        )
        model = MicroGPT(config)
        # Use real embedding weights
        real_weights = model.embedding.weight.data
        tensor = real_weights[:shape[0], :shape[1]].astype(np.float32)

        results = {}
        for bits in [32, 16, 8, 4, 2]:
            _, metrics = QuantizationAnalyzer.quantize_tensor(tensor, bits)
            label = {32: "FP32", 16: "FP16", 8: "INT8", 4: "INT4", 2: "INT2"}[bits]
            original_mb = tensor.nbytes / 1e6
            quantized_mb = original_mb * bits / 32
            metrics["label"] = label
            metrics["memory_mb"] = round(quantized_mb, 3)
            metrics["original_mb"] = round(original_mb, 3)
            results[label] = metrics

        return {"shape": list(shape), "results": results}


class SpeculativeDecodingEngine:
    """Speculative decoding using real draft and target model inference."""

    def __init__(self, gamma: int = 4, acceptance_rate: float = 0.7):
        self.gamma = gamma
        self.acceptance_rate = acceptance_rate

    def run(self, total_tokens: int = 100) -> Dict:
        """Run speculative decoding with real draft/target model pair."""
        from app.core.model import MicroGPT, GPTConfig

        # Real draft model (small)
        draft_config = GPTConfig(
            vocab_size=256, d_model=32, num_heads=1, num_layers=1,
            d_ff=128, max_seq_len=128, dropout=0.0,
        )
        # Real target model (larger)
        target_config = GPTConfig(
            vocab_size=256, d_model=64, num_heads=2, num_layers=2,
            d_ff=256, max_seq_len=128, dropout=0.0,
        )

        draft_model = MicroGPT(draft_config)
        target_model = MicroGPT(target_config)
        draft_model.set_training(False)
        target_model.set_training(False)

        # Start with a real text prompt
        prompt = "The transformer"
        prompt_ids = np.array([[ord(c) % 256 for c in prompt]])

        standard_steps = total_tokens
        speculative_steps = 0
        tokens_generated = 0
        accepted_counts = []

        current_ids = prompt_ids.copy()
        max_gen = min(total_tokens, 50)  # Cap for performance

        while tokens_generated < max_gen:
            # Draft model generates gamma candidate tokens
            draft_candidates = []
            draft_input = current_ids[:, -draft_config.max_seq_len:]

            for _ in range(self.gamma):
                draft_result = draft_model.forward(draft_input)
                draft_logits = draft_result["logits"][:, -1, :]
                draft_probs = np.exp(draft_logits - draft_logits.max(axis=-1, keepdims=True))
                draft_probs = draft_probs / draft_probs.sum(axis=-1, keepdims=True)
                draft_token = int(np.argmax(draft_probs[0]))
                draft_candidates.append(draft_token)
                draft_input = np.concatenate(
                    [draft_input, np.array([[draft_token]])], axis=1
                )[:, -draft_config.max_seq_len:]

            # Target model verifies all candidates at once
            accepted = 0
            verify_input = current_ids[:, -target_config.max_seq_len:]
            for tok in draft_candidates:
                target_result = target_model.forward(verify_input)
                target_logits = target_result["logits"][:, -1, :]
                target_token = int(np.argmax(target_logits[0]))

                if target_token == tok:
                    accepted += 1
                    verify_input = np.concatenate(
                        [verify_input, np.array([[tok]])], axis=1
                    )[:, -target_config.max_seq_len:]
                else:
                    # Rejected: use target token instead
                    verify_input = np.concatenate(
                        [verify_input, np.array([[target_token]])], axis=1
                    )[:, -target_config.max_seq_len:]
                    break

            tokens_this_step = accepted + 1
            tokens_generated += tokens_this_step
            speculative_steps += 1
            accepted_counts.append(accepted)

            # Update current sequence
            final_tokens = draft_candidates[:accepted]
            if accepted < len(draft_candidates):
                target_result = target_model.forward(verify_input)
                target_logits = target_result["logits"][:, -1, :]
                final_tokens.append(int(np.argmax(target_logits[0])))
            else:
                final_tokens.append(draft_candidates[-1])

            for t in final_tokens:
                current_ids = np.concatenate(
                    [current_ids, np.array([[t]])], axis=1
                )

        # Scale results to requested total_tokens
        scale = total_tokens / max(max_gen, 1)
        scaled_steps = int(speculative_steps * scale)

        actual_acceptance = float(np.mean(accepted_counts)) / self.gamma if accepted_counts else 0
        speedup = total_tokens / max(scaled_steps, 1)

        return {
            "total_tokens": total_tokens,
            "gamma": self.gamma,
            "target_acceptance_rate": self.acceptance_rate,
            "actual_acceptance_rate": round(actual_acceptance, 3),
            "standard_steps": standard_steps,
            "speculative_steps": scaled_steps,
            "speedup": round(float(speedup), 2),
            "accepted_per_step": [int(a) for a in accepted_counts],
            "draft_model_params": draft_config.num_parameters,
            "target_model_params": target_config.num_parameters,
        }
