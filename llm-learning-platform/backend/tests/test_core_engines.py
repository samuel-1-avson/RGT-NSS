"""Tests for the RLHF, LoRA, Evaluation, and other core engines."""

import pytest
import numpy as np


class TestRLHF:
    def test_reward_model_score(self):
        from app.core.rlhf import RewardModel
        rm = RewardModel()
        score = rm.score("This is a great response")
        assert isinstance(score, float)

    def test_reward_model_train(self):
        from app.core.rlhf import RewardModel, PreferencePair
        rm = RewardModel()
        pairs = [
            PreferencePair(chosen="Good answer", rejected="Bad answer"),
            PreferencePair(chosen="Helpful response", rejected="Unhelpful"),
        ]
        result = rm.train_on_preferences(pairs, num_epochs=2)
        assert "losses" in result
        assert len(result["losses"]) == 2

    def test_ppo_train(self):
        from app.core.rlhf import PPOTrainer, RLHFConfig
        config = RLHFConfig()
        trainer = PPOTrainer(config)
        result = trainer.train(num_steps=5, num_responses=2)
        assert len(result) == 5

    def test_dpo_train(self):
        from app.core.rlhf import DPOTrainer
        trainer = DPOTrainer(beta=0.1)
        result = trainer.train(num_steps=5)
        assert len(result) == 5


class TestLoRA:
    def test_lora_layer(self):
        from app.core.lora import LoRALayer, LoRAConfig
        config = LoRAConfig(d_model=64, rank=4)
        layer = LoRALayer(config)
        x = np.random.randn(2, 64)
        out = layer.forward(x)
        assert out.shape == (2, 64)

    def test_lora_model(self):
        from app.core.lora import LoRAModel, LoRAConfig
        config = LoRAConfig(d_model=64, rank=4, num_layers=2)
        model = LoRAModel(config)
        summary = model.get_summary()
        assert summary["total_params"] > 0
        assert summary["trainable_params"] > 0
        assert summary["trainable_pct"] < 100

    def test_quantizer(self):
        from app.core.lora import QLoRAQuantizer
        q = QLoRAQuantizer()
        matrix = np.random.randn(32, 32).astype(np.float32)
        analysis = q.analyze(matrix)
        assert "original_mb" in analysis
        assert "quantized_mb" in analysis

    def test_compare_peft(self):
        from app.core.lora import compare_peft_methods
        result = compare_peft_methods(64, 2)
        assert len(result) > 0
        assert "method" in result[0]


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
        assert "rouge_1" in result
        assert "rouge_l" in result

    def test_perplexity(self):
        from app.core.evaluation import compute_perplexity
        result = compute_perplexity([2.5, 2.0, 1.5])
        assert "perplexity" in result
        assert result["perplexity"] > 0

    def test_benchmark(self):
        from app.core.evaluation import run_benchmark_suite
        result = run_benchmark_suite("test-model")
        assert "benchmarks" in result
        assert len(result["benchmarks"]) > 0


class TestInferenceOpt:
    def test_kv_cache(self):
        from app.core.inference_opt import KVCacheAnalyzer, KVCacheConfig
        sim = KVCacheAnalyzer(KVCacheConfig(num_layers=4, num_heads=4, head_dim=32))
        result = sim.analyze_generation(prompt_len=5, gen_len=10)
        assert len(result) > 0
        assert result[-1]["cache_mb"] > 0

    def test_quantization(self):
        from app.core.inference_opt import QuantizationAnalyzer
        analyzer = QuantizationAnalyzer()
        result = analyzer.compare_formats(64, 64)
        assert len(result) > 0

    def test_speculative_decoding(self):
        from app.core.inference_opt import SpeculativeDecodingEngine
        sim = SpeculativeDecodingEngine(gamma=4, acceptance_rate=0.7)
        result = sim.run(total_tokens=20)
        assert "speedup" in result


class TestInterpretability:
    def test_logit_lens(self):
        from app.core.interpretability import LogitLens
        ll = LogitLens(num_layers=4, vocab_size=100)
        result = ll.analyze("test text", top_k=3)
        assert "layers" in result
        assert len(result["layers"]) == 4

    def test_neuron_analyzer(self):
        from app.core.interpretability import NeuronAnalyzer
        na = NeuronAnalyzer(hidden_size=64)
        result = na.analyze("test", layer=0, top_k=5)
        assert "top_neurons" in result
        assert "dead_neurons" in result


class TestDistributed:
    def test_data_parallel(self):
        from app.core.distributed import DataParallelAnalyzer, DistributedConfig
        sim = DataParallelAnalyzer(DistributedConfig(num_gpus=4, model_params_m=100, batch_size=32))
        result = sim.analyze()
        assert "effective_batch_size" in result

    def test_compare_strategies(self):
        from app.core.distributed import compare_strategies, DistributedConfig
        result = compare_strategies(DistributedConfig(num_gpus=4, model_params_m=100, batch_size=32, gpu_memory_gb=24))
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
        assert "detected_techniques" in result
        assert "chain_of_thought" in result["detected_techniques"]

    def test_render_template(self):
        from app.core.prompt_eng import render_template
        result = render_template("zero_shot", {"question": "What is 2+2?"})
        assert "rendered" in result


class TestSafety:
    def test_evaluate(self):
        from app.core.safety import evaluate_safety
        result = evaluate_safety("This is a normal text.")
        assert "safe" in result
        assert "overall_score" in result
        assert "categories" in result

    def test_redteam(self):
        from app.core.safety import run_redteam_suite
        result = run_redteam_suite()
        assert "results" in result
        assert len(result["results"]) > 0

    def test_constitutional(self):
        from app.core.safety import apply_constitutional_ai
        result = apply_constitutional_ai("Here is a response to review.")
        assert "reviews" in result
        assert len(result["reviews"]) > 0


class TestLongContext:
    def test_rope(self):
        from app.core.long_context import RoPEAnalyzer
        analyzer = RoPEAnalyzer(dim=64)
        result = analyzer.compute_frequencies(max_position=128, method="none")
        assert "frequencies" in result

    def test_alibi(self):
        from app.core.long_context import ALiBiAnalyzer
        analyzer = ALiBiAnalyzer(num_heads=4)
        result = analyzer.compute_bias_matrix(seq_len=8)
        assert "slopes" in result
        assert "bias_matrix_sample" in result

    def test_compare_methods(self):
        from app.core.long_context import compare_position_methods
        result = compare_position_methods(64, 4, 128)
        assert "methods" in result
        assert len(result["methods"]) > 0
