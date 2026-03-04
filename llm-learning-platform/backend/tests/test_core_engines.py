"""Tests for the RLHF, LoRA, Evaluation, and other core engines.

Fixed to match actual API signatures of the core engine implementations.
"""

import pytest
import numpy as np


class TestRLHF:
    def test_reward_model_score(self):
        from app.core.rlhf import RewardModel
        rm = RewardModel()
        token_ids = [ord(c) % 256 for c in "This is a great response"]
        score = rm.score(token_ids)
        assert isinstance(score, float)

    def test_reward_model_train(self):
        from app.core.rlhf import RewardModel, PreferencePair
        rm = RewardModel()
        pairs = [
            PreferencePair(prompt="Question?", chosen="Good answer", rejected="Bad answer"),
            PreferencePair(prompt="Another?", chosen="Helpful response", rejected="Unhelpful"),
        ]
        # Returns List[Dict] with epoch, loss, accuracy
        result = rm.train_on_preferences(pairs, lr=1e-3, epochs=2)
        assert isinstance(result, list)
        assert len(result) == 2
        assert "epoch" in result[0]
        assert "loss" in result[0]
        assert "accuracy" in result[0]

    def test_ppo_train(self):
        from app.core.rlhf import PPOTrainer, RLHFConfig
        config = RLHFConfig()
        trainer = PPOTrainer(config)
        result = trainer.train(num_steps=5, num_responses=2)
        assert isinstance(result, list)
        assert len(result) == 5
        # Returns RLHFStepResult dataclass, use hasattr
        assert hasattr(result[0], "step")
        assert hasattr(result[0], "policy_loss")

    def test_dpo_train(self):
        from app.core.rlhf import DPOTrainer
        trainer = DPOTrainer(beta=0.1)
        result = trainer.train(num_steps=5)
        assert isinstance(result, list)
        assert len(result) == 5
        assert "step" in result[0]


class TestLoRA:
    def test_lora_layer(self):
        from app.core.lora import LoRALayer
        import torch
        layer = LoRALayer(in_features=64, out_features=64, rank=4, alpha=8)
        x = torch.randn(2, 64)
        out = layer.forward(x)
        assert out.shape == (2, 64)

    def test_lora_layer_merge(self):
        from app.core.lora import LoRALayer
        layer = LoRALayer(in_features=64, out_features=64, rank=4)
        merged = layer.merge_weights()
        assert merged.shape == (64, 64)

    def test_lora_layer_info(self):
        from app.core.lora import LoRALayer
        layer = LoRALayer(in_features=64, out_features=64, rank=4, alpha=8)
        info = layer.get_info("test_layer")
        assert info.lora_params > 0
        assert info.compression_ratio < 1.0

    def test_lora_model(self):
        from app.core.lora import LoRAModel, LoRAConfig
        config = LoRAConfig(rank=4, alpha=8)
        model = LoRAModel(d_model=64, num_layers=2, config=config)
        summary = model.get_summary()
        assert summary["total_lora_params"] > 0
        assert summary["total_original_params"] > 0
        assert summary["param_percentage"] < 100

    def test_lora_model_forward(self):
        from app.core.lora import LoRAModel, LoRAConfig
        config = LoRAConfig(rank=4, alpha=8)
        model = LoRAModel(d_model=64, num_layers=2, config=config)
        x = np.random.randn(2, 64).astype(np.float32)
        out = model.forward(x)
        assert out.shape == (2, 64)

    def test_lora_train(self):
        from app.core.lora import LoRAModel, LoRAConfig
        config = LoRAConfig(rank=4, alpha=8)
        model = LoRAModel(d_model=64, num_layers=2, config=config)
        results = model.train(num_steps=5)
        assert len(results) == 5
        assert "loss" in results[0]
        assert "accuracy" in results[0]

    def test_quantizer(self):
        from app.core.lora import QLoRAQuantizer
        q = QLoRAQuantizer()
        matrix = np.random.randn(32, 32).astype(np.float32)
        analysis = q.analyze_quantization(matrix)
        assert "original_bytes" in analysis
        assert "quantized_bytes" in analysis
        assert "compression_ratio" in analysis
        assert analysis["compression_ratio"] > 1.0

    def test_quantizer_roundtrip(self):
        from app.core.lora import QLoRAQuantizer
        q = QLoRAQuantizer()
        matrix = np.random.randn(32, 32).astype(np.float32)
        indices, absmax = q.quantize(matrix)
        restored = q.dequantize(indices, absmax, matrix.shape)
        assert restored.shape == matrix.shape
        assert np.mean(np.abs(matrix - restored)) < 0.5

    def test_compare_peft(self):
        from app.core.lora import compare_peft_methods
        result = compare_peft_methods(64, 2)
        assert "methods" in result
        assert "d_model" in result
        assert len(result["methods"]) > 0
        assert "full_finetuning" in result["methods"]


class TestEvaluation:
    def test_bleu(self):
        from app.core.evaluation import compute_bleu
        result = compute_bleu("the cat sat on the mat", "the cat is on the mat")
        assert "bleu" in result
        assert 0 <= result["bleu"] <= 1

    def test_rouge(self):
        from app.core.evaluation import compute_rouge
        result = compute_rouge(
            "the quick brown fox jumps over the lazy dog",
            "the fast brown fox leaps over the lazy dog",
        )
        # Actual keys: rouge1, rouge2, rougeL
        assert "rouge1" in result
        assert "rougeL" in result

    def test_perplexity(self):
        from app.core.evaluation import compute_perplexity
        # Returns a float, not a dict
        result = compute_perplexity([2.5, 2.0, 1.5])
        assert isinstance(result, float)
        assert result > 0

    def test_benchmark(self):
        from app.core.evaluation import run_benchmark_suite
        result = run_benchmark_suite("test-model")
        assert "benchmarks" in result
        assert isinstance(result["benchmarks"], dict)


class TestInferenceOpt:
    def test_kv_cache(self):
        from app.core.inference_opt import KVCacheAnalyzer, KVCacheConfig
        config = KVCacheConfig(num_layers=4, num_heads=4, head_dim=32)
        analyzer = KVCacheAnalyzer(config)
        result = analyzer.analyze_generation(prompt_len=5, gen_len=10)
        assert len(result) > 0
        assert result[-1]["cache_mb"] > 0

    def test_quantization(self):
        from app.core.inference_opt import QuantizationAnalyzer
        result = QuantizationAnalyzer.compare_quantizations(shape=(64, 64))
        assert "results" in result
        assert len(result["results"]) > 0

    def test_speculative_decoding(self):
        from app.core.inference_opt import SpeculativeDecodingEngine
        engine = SpeculativeDecodingEngine(gamma=4, acceptance_rate=0.7)
        result = engine.run(total_tokens=20)
        assert "speedup" in result


class TestInterpretability:
    def test_logit_lens(self):
        from app.core.interpretability import LogitLens
        ll = LogitLens(num_layers=4, vocab_size=100)
        # Actual method: probe_all_layers, not analyze
        result = ll.probe_all_layers("test text", top_k=3)
        assert "layers" in result
        assert len(result["layers"]) == 4

    def test_neuron_analyzer(self):
        from app.core.interpretability import NeuronAnalyzer
        # Init takes d_model, d_ff — not hidden_size
        na = NeuronAnalyzer(d_model=64, d_ff=256)
        # Actual method: analyze_neurons, not analyze
        result = na.analyze_neurons("test", layer=0, top_k=5)
        assert "top_neurons" in result
        assert "dead_neurons" in result


class TestDistributed:
    def test_data_parallel(self):
        from app.core.distributed import DataParallelAnalyzer, DistributedConfig
        config = DistributedConfig(num_gpus=4, model_params_m=100, batch_size=32)
        analyzer = DataParallelAnalyzer(config)
        result = analyzer.analyze()
        assert "effective_batch_size" in result

    def test_compare_strategies(self):
        from app.core.distributed import compare_strategies, DistributedConfig
        config = DistributedConfig(num_gpus=4, model_params_m=100, batch_size=32, gpu_memory_gb=24)
        result = compare_strategies(config)
        assert len(result) > 0


class TestPromptEng:
    def test_list_templates(self):
        from app.core.prompt_eng import list_templates
        templates = list_templates()
        assert len(templates) > 0

    def test_analyze_prompt(self):
        from app.core.prompt_eng import analyze_prompt
        result = analyze_prompt("Let's think step by step. What is 2+2?")
        assert "word_count" in result
        # Actual key: techniques_detected, not detected_techniques
        assert "techniques_detected" in result
        assert result["techniques_detected"]["chain_of_thought"] is True

    def test_render_template(self):
        from app.core.prompt_eng import render_template
        result = render_template("zero_shot", {"question": "What is 2+2?"})
        assert "rendered" in result


class TestSafety:
    def test_evaluate(self):
        from app.core.safety import evaluate_safety
        result = evaluate_safety("This is a normal text.")
        # Actual keys: is_safe, safety_score, flags
        assert "is_safe" in result
        assert "safety_score" in result
        assert "flags" in result

    def test_redteam(self):
        from app.core.safety import run_redteam_suite
        result = run_redteam_suite()
        assert "results" in result
        assert len(result["results"]) > 0

    def test_constitutional(self):
        from app.core.safety import apply_constitutional_ai
        result = apply_constitutional_ai("Here is a response to review.")
        # Actual keys: critiques, average_score, needs_revision
        assert "critiques" in result
        assert "average_score" in result
        assert len(result["critiques"]) > 0


class TestLongContext:
    def test_rope(self):
        from app.core.long_context import RoPEAnalyzer, RoPEConfig
        config = RoPEConfig(dim=64)
        analyzer = RoPEAnalyzer(config)
        result = analyzer.compute_frequencies(scaling="none")
        assert "frequency_values" in result

    def test_alibi(self):
        from app.core.long_context import ALiBiAnalyzer
        analyzer = ALiBiAnalyzer(num_heads=4)
        result = analyzer.compute_bias_matrix(seq_len=8)
        assert "slopes" in result
        assert "heads" in result

    def test_compare_methods(self):
        from app.core.long_context import compare_position_methods
        result = compare_position_methods(seq_len=128, dim=64, num_heads=4)
        assert "methods" in result
        assert len(result["methods"]) > 0
