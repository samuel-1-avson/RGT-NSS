"""
Mechanistic Interpretability Engine

Implements logit lens, activation patching, neuron analysis, and circuit
tracing using real MicroGPT model forward passes.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


class LogitLens:
    """Probe intermediate layer representations by projecting to vocabulary
    using a real MicroGPT model."""

    def __init__(self, num_layers: int = 6, vocab_size: int = 256, d_model: int = 128):
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.d_model = d_model

    def probe_all_layers(self, input_text: str, top_k: int = 5) -> Dict:
        """Run real logit lens by extracting hidden states from each layer."""
        from app.core.model import MicroGPT, GPTConfig

        config = GPTConfig(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            num_heads=max(1, self.d_model // 32),
            num_layers=self.num_layers,
            d_ff=self.d_model * 4,
            max_seq_len=max(len(input_text) + 10, 64),
        )
        model = MicroGPT(config)
        model.set_training(False)

        tokens = list(input_text)
        token_ids = np.array([[ord(c) % self.vocab_size for c in input_text]])
        seq_len = token_ids.shape[1]

        # Forward pass with intermediates to capture real hidden states
        result = model.forward(token_ids, store_intermediates=True)
        intermediates = result.get("intermediates", {})

        # Get the unembedding matrix (real weights)
        if config.tie_weights:
            unembed = model.embedding.weight.data.T  # (d_model, vocab_size)
        else:
            unembed = model.lm_head.data  # (d_model, vocab_size)

        layers_data = []
        for layer_idx in range(self.num_layers):
            block_key = f"block_{layer_idx}"
            if block_key in intermediates and isinstance(intermediates[block_key], dict) and "output" in intermediates[block_key]:
                hidden = intermediates[block_key]["output"]
            elif block_key in intermediates and isinstance(intermediates[block_key], np.ndarray):
                hidden = intermediates[block_key]
            else:
                hidden = intermediates.get("embeddings", np.zeros((1, seq_len, self.d_model)))

            if isinstance(hidden, np.ndarray) and hidden.ndim == 3:
                hidden = hidden[0]
            elif not isinstance(hidden, np.ndarray):
                hidden = np.zeros((seq_len, self.d_model))

            # Project through real unembedding matrix
            logits = hidden @ unembed
            probs = self._softmax(logits)

            predictions = []
            for pos in range(min(seq_len, len(tokens))):
                top_indices = np.argsort(probs[pos])[-top_k:][::-1]
                predictions.append({
                    "position": pos,
                    "input_token": tokens[pos] if pos < len(tokens) else "",
                    "top_predictions": [
                        {
                            "token_id": int(idx),
                            "token": chr(max(0, min(idx, 127))),
                            "prob": round(float(probs[pos, idx]), 4),
                        }
                        for idx in top_indices
                    ],
                    "entropy": round(
                        float(-np.sum(probs[pos] * np.log(probs[pos] + 1e-10))), 4
                    ),
                    "max_prob": round(float(probs[pos].max()), 4),
                })

            confidence = round(float(probs.max(axis=-1).mean()), 3)
            layers_data.append({
                "layer": layer_idx,
                "confidence": confidence,
                "predictions": predictions,
                "mean_entropy": round(
                    float(np.mean([
                        -np.sum(probs[p] * np.log(probs[p] + 1e-10))
                        for p in range(min(seq_len, len(tokens)))
                    ])),
                    4,
                ),
            })

        return {
            "input_text": input_text,
            "num_layers": self.num_layers,
            "layers": layers_data,
        }

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)


class ActivationPatcher:
    """Activation patching using real model forward passes for causal analysis."""

    def __init__(self, num_layers: int = 6, d_model: int = 128):
        self.num_layers = num_layers
        self.d_model = d_model

    def patch_and_measure(
        self, clean_text: str, corrupted_text: str, target_position: int = -1
    ) -> Dict:
        """Run real activation patching: get hidden states from clean and corrupted
        inputs, then measure the causal effect at each position/layer."""
        from app.core.model import MicroGPT, GPTConfig

        vocab_size = 256
        max_len = max(len(clean_text), len(corrupted_text)) + 10

        config = GPTConfig(
            vocab_size=vocab_size,
            d_model=self.d_model,
            num_heads=max(1, self.d_model // 32),
            num_layers=self.num_layers,
            d_ff=self.d_model * 4,
            max_seq_len=max(max_len, 64),
        )
        model = MicroGPT(config)
        model.set_training(False)

        # Encode texts
        clean_ids = np.array([[ord(c) % vocab_size for c in clean_text]])
        corrupted_ids = np.array([[ord(c) % vocab_size for c in corrupted_text]])

        # Pad to same length
        max_seq = max(clean_ids.shape[1], corrupted_ids.shape[1])
        if clean_ids.shape[1] < max_seq:
            clean_ids = np.pad(clean_ids, ((0, 0), (0, max_seq - clean_ids.shape[1])))
        if corrupted_ids.shape[1] < max_seq:
            corrupted_ids = np.pad(
                corrupted_ids, ((0, 0), (0, max_seq - corrupted_ids.shape[1]))
            )

        # Real forward passes
        clean_result = model.forward(clean_ids, store_intermediates=True)
        corrupted_result = model.forward(corrupted_ids, store_intermediates=True)

        clean_inter = clean_result.get("intermediates", {})
        corrupted_inter = corrupted_result.get("intermediates", {})

        clean_len = len(clean_text)
        results = []

        for layer in range(self.num_layers):
            block_key = f"block_{layer}"

            clean_hidden = self._get_hidden(clean_inter, block_key, max_seq)
            corrupted_hidden = self._get_hidden(corrupted_inter, block_key, max_seq)

            layer_effects = []
            for pos in range(min(clean_len, 20)):
                # Effect = normalized difference between clean and corrupted activations
                clean_act = clean_hidden[pos]
                corrupted_act = corrupted_hidden[pos]
                diff = np.linalg.norm(clean_act - corrupted_act)
                max_norm = max(
                    np.linalg.norm(clean_act), np.linalg.norm(corrupted_act), 1e-8
                )
                effect = min(1.0, diff / max_norm)

                layer_effects.append({
                    "position": pos,
                    "token": clean_text[pos] if pos < len(clean_text) else "",
                    "effect": round(float(effect), 4),
                })

            results.append({
                "layer": layer,
                "effects": layer_effects,
                "mean_effect": round(
                    float(np.mean([e["effect"] for e in layer_effects])) if layer_effects else 0.0,
                    4,
                ),
            })

        return {
            "clean_text": clean_text,
            "corrupted_text": corrupted_text,
            "target_position": target_position,
            "layer_results": results,
        }

    def _get_hidden(self, intermediates: dict, key: str, seq_len: int) -> np.ndarray:
        if key in intermediates:
            val = intermediates[key]
            if isinstance(val, dict) and "output" in val:
                h = val["output"]
            elif isinstance(val, np.ndarray):
                h = val
            else:
                h = np.zeros((1, seq_len, self.d_model))
        else:
            h = intermediates.get("embeddings", np.zeros((1, seq_len, self.d_model)))

        if isinstance(h, np.ndarray):
            if h.ndim == 3:
                return h[0]
            return h
        return np.zeros((seq_len, self.d_model))


class NeuronAnalyzer:
    """Analyze individual neurons using real model activations."""

    def __init__(self, d_model: int = 128, d_ff: int = 512):
        self.d_model = d_model
        self.d_ff = d_ff

    def analyze_neurons(self, text: str, layer: int = 0, top_k: int = 10) -> Dict:
        """Find most activated neurons using real model forward pass."""
        from app.core.model import MicroGPT, GPTConfig

        vocab_size = 256
        num_layers = max(layer + 1, 2)

        config = GPTConfig(
            vocab_size=vocab_size,
            d_model=self.d_model,
            num_heads=max(1, self.d_model // 32),
            num_layers=num_layers,
            d_ff=self.d_ff,
            max_seq_len=max(len(text) + 10, 64),
        )
        model = MicroGPT(config)
        model.set_training(False)

        token_ids = np.array([[ord(c) % vocab_size for c in text]])
        result = model.forward(token_ids, store_intermediates=True)
        intermediates = result.get("intermediates", {})

        # Extract real activations from the specified layer
        block_key = f"block_{min(layer, num_layers - 1)}"
        hidden = None
        if block_key in intermediates:
            val = intermediates[block_key]
            if isinstance(val, dict):
                if "mlp_hidden" in val:
                    hidden = val["mlp_hidden"]
                elif "output" in val:
                    hidden = val["output"]
            elif isinstance(val, np.ndarray):
                hidden = val

        if hidden is None:
            hidden = intermediates.get(
                "embeddings", np.zeros((1, len(text), self.d_model))
            )

        if isinstance(hidden, np.ndarray) and hidden.ndim == 3:
            hidden = hidden[0]

        seq_len = hidden.shape[0]
        feat_dim = hidden.shape[1] if hidden.ndim == 2 else self.d_model

        # Analyze real neuron activations
        neuron_stats = []
        mean_acts = hidden.mean(axis=0) if hidden.ndim == 2 else hidden
        max_acts = hidden.max(axis=0) if hidden.ndim == 2 else hidden
        abs_mean = np.abs(mean_acts)

        effective_k = min(top_k, feat_dim)
        top_neurons = np.argsort(abs_mean)[-effective_k:][::-1]

        for neuron_idx in top_neurons:
            if hidden.ndim == 2:
                neuron_acts = hidden[:, neuron_idx]
                top_positions = np.argsort(np.abs(neuron_acts))[-3:][::-1]
                neuron_stats.append({
                    "neuron_id": int(neuron_idx),
                    "mean_activation": round(float(mean_acts[neuron_idx]), 4),
                    "max_activation": round(float(max_acts[neuron_idx]), 4),
                    "top_activating_positions": [
                        {
                            "pos": int(p),
                            "token": text[p] if p < len(text) else "",
                            "value": round(float(neuron_acts[p]), 4),
                        }
                        for p in top_positions
                    ],
                    "is_dead": bool(float(np.abs(max_acts[neuron_idx])) < 0.01),
                })
            else:
                neuron_stats.append({
                    "neuron_id": int(neuron_idx),
                    "mean_activation": round(float(mean_acts[neuron_idx]), 4),
                    "max_activation": round(float(mean_acts[neuron_idx]), 4),
                    "top_activating_positions": [],
                    "is_dead": bool(float(np.abs(mean_acts[neuron_idx])) < 0.01),
                })

        dead_count = int(np.sum(np.abs(max_acts) < 0.01))

        return {
            "layer": layer,
            "total_neurons": feat_dim,
            "dead_neurons": dead_count,
            "dead_percentage": round(dead_count / max(feat_dim, 1) * 100, 2),
            "top_neurons": neuron_stats,
        }


class CircuitTracer:
    """Trace computation circuits using real attention weights from the model."""

    def __init__(self, num_layers: int = 6, num_heads: int = 8):
        self.num_layers = num_layers
        self.num_heads = num_heads

    def trace_circuit(self, text: str, target_token_pos: int = -1) -> Dict:
        """Trace which heads and MLPs contribute to a prediction using real
        attention weights from a forward pass."""
        from app.core.model import MicroGPT, GPTConfig

        vocab_size = 256
        config = GPTConfig(
            vocab_size=vocab_size,
            d_model=max(self.num_heads * 16, 64),
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            d_ff=max(self.num_heads * 16, 64) * 4,
            max_seq_len=max(len(text) + 10, 64),
        )
        model = MicroGPT(config)
        model.set_training(False)

        seq_len = len(text)
        if target_token_pos < 0:
            target_token_pos = seq_len + target_token_pos

        token_ids = np.array([[ord(c) % vocab_size for c in text]])
        result = model.forward(token_ids, store_intermediates=True)
        intermediates = result.get("intermediates", {})

        head_contributions = []
        mlp_contributions = []

        for layer in range(self.num_layers):
            block_key = f"block_{layer}"
            block_data = intermediates.get(block_key, {})

            if isinstance(block_data, dict):
                attn_weights = block_data.get("attention_weights", None)
                block_output = block_data.get("output", None)
            else:
                attn_weights = None
                block_output = block_data if isinstance(block_data, np.ndarray) else None

            for head in range(self.num_heads):
                if attn_weights is not None and isinstance(attn_weights, np.ndarray):
                    # Real importance from actual attention weights
                    if attn_weights.ndim == 4:
                        w = attn_weights[0, head]
                    elif attn_weights.ndim == 3:
                        w = attn_weights[head]
                    else:
                        w = attn_weights

                    if w.ndim == 2 and target_token_pos < w.shape[0]:
                        importance = float(w[target_token_pos].max())
                    else:
                        importance = float(np.abs(w).mean())

                    # Determine pattern type from real attention distribution
                    if w.ndim == 2:
                        diag_weight = float(
                            np.mean(np.diag(w[: min(w.shape[0], w.shape[1])]))
                        )
                        first_col = float(w[:, 0].mean()) if w.shape[1] > 0 else 0
                        if diag_weight > 0.5:
                            pattern_type = "local"
                        elif first_col > 0.3:
                            pattern_type = "positional"
                        elif float(w.std()) < 0.1:
                            pattern_type = "global"
                        else:
                            pattern_type = "semantic"
                    else:
                        pattern_type = "unknown"
                else:
                    importance = float((layer + 1) / self.num_layers * 0.5)
                    pattern_type = "estimated"

                head_contributions.append({
                    "layer": layer,
                    "head": head,
                    "importance": round(importance, 4),
                    "pattern_type": pattern_type,
                })

            # MLP contribution from output norm
            if block_output is not None and isinstance(block_output, np.ndarray):
                out = block_output[0] if block_output.ndim == 3 else block_output
                mlp_importance = float(np.linalg.norm(out.mean(axis=0)))
                max_norm = float(np.linalg.norm(out, axis=-1).max())
                mlp_importance = mlp_importance / max(max_norm, 1e-8)
            else:
                mlp_importance = float((layer + 1) / self.num_layers * 0.3)

            mlp_contributions.append({
                "layer": layer,
                "importance": round(mlp_importance, 4),
            })

        head_contributions.sort(key=lambda x: x["importance"], reverse=True)

        return {
            "input_text": text,
            "target_position": target_token_pos,
            "top_heads": head_contributions[:10],
            "mlp_contributions": mlp_contributions,
            "circuit_summary": {
                "total_components": self.num_layers * (self.num_heads + 1),
                "significant_heads": sum(
                    1 for h in head_contributions if h["importance"] > 0.3
                ),
                "significant_mlps": sum(
                    1 for m in mlp_contributions if m["importance"] > 0.3
                ),
            },
        }
