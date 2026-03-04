"""Tests for all API routers using FastAPI TestClient."""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


class TestHealthAPI:
    def test_health(self):
        r = client.get("/api/v1/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "healthy"


class TestTokenizationAPI:
    def test_tokenize(self):
        r = client.post("/api/v1/tokenization/tokenize", json={
            "text": "Hello world",
            "strategy": "character",
        })
        assert r.status_code == 200
        data = r.json()
        assert "tokens" in data or "ids" in data

    def test_compare(self):
        r = client.post("/api/v1/tokenization/compare", json={
            "text": "Hello world",
            "strategies": ["character", "bpe"],
        })
        assert r.status_code == 200


class TestRLHFAPI:
    def test_reward_score(self):
        r = client.post("/api/v1/rlhf/reward/score", json={
            "text": "This is a test response.",
        })
        assert r.status_code == 200
        data = r.json()
        assert "score" in data

    def test_ppo_train(self):
        r = client.post("/api/v1/rlhf/ppo/train", json={
            "num_steps": 3,
            "num_responses": 2,
        })
        assert r.status_code == 200
        data = r.json()
        assert "steps" in data

    def test_dpo_train(self):
        r = client.post("/api/v1/rlhf/dpo/train", json={
            "num_steps": 3,
            "beta": 0.1,
        })
        assert r.status_code == 200
        data = r.json()
        assert "steps" in data

    def test_methods(self):
        r = client.get("/api/v1/rlhf/methods")
        assert r.status_code == 200


class TestLoRAAPI:
    def test_create(self):
        r = client.post("/api/v1/lora/create", json={
            "d_model": 64,
            "rank": 4,
            "num_layers": 2,
        })
        assert r.status_code == 200

    def test_ranks(self):
        r = client.get("/api/v1/lora/ranks")
        assert r.status_code == 200

    def test_compare_peft(self):
        r = client.post("/api/v1/lora/compare-peft", json={
            "d_model": 64,
            "num_layers": 2,
        })
        assert r.status_code == 200


class TestEvaluationAPI:
    def test_bleu(self):
        r = client.post("/api/v1/evaluation/bleu", json={
            "reference": "the cat sat on the mat",
            "hypothesis": "the cat is on the mat",
        })
        assert r.status_code == 200
        data = r.json()
        assert "bleu" in data

    def test_rouge(self):
        r = client.post("/api/v1/evaluation/rouge", json={
            "reference": "the quick brown fox",
            "hypothesis": "the fast brown fox",
        })
        assert r.status_code == 200

    def test_perplexity(self):
        r = client.post("/api/v1/evaluation/perplexity", json={
            "losses": [2.5, 2.0, 1.5],
        })
        assert r.status_code == 200

    def test_metrics(self):
        r = client.get("/api/v1/evaluation/metrics")
        assert r.status_code == 200


class TestInferenceOptAPI:
    def test_kv_cache(self):
        r = client.post("/api/v1/inference-opt/kv-cache/analyze", json={
            "num_layers": 4,
            "num_heads": 4,
            "head_dim": 32,
            "prompt_len": 5,
            "gen_len": 10,
        })
        assert r.status_code == 200

    def test_quantization(self):
        r = client.post("/api/v1/inference-opt/quantization/compare", json={
            "rows": 64,
            "cols": 64,
        })
        assert r.status_code == 200

    def test_techniques(self):
        r = client.get("/api/v1/inference-opt/techniques")
        assert r.status_code == 200


class TestInterpretabilityAPI:
    def test_logit_lens(self):
        r = client.post("/api/v1/interpretability/logit-lens", json={
            "text": "The capital of France",
            "num_layers": 4,
            "top_k": 3,
        })
        assert r.status_code == 200

    def test_neurons(self):
        r = client.post("/api/v1/interpretability/neurons", json={
            "text": "Hello",
            "layer": 0,
            "top_k": 5,
        })
        assert r.status_code == 200

    def test_tools(self):
        r = client.get("/api/v1/interpretability/tools")
        assert r.status_code == 200


class TestDistributedAPI:
    def test_data_parallel(self):
        r = client.post("/api/v1/distributed/data-parallel", json={
            "num_gpus": 4,
            "model_params_m": 100,
            "batch_size": 32,
            "gpu_memory_gb": 24,
        })
        assert r.status_code == 200

    def test_compare_all(self):
        r = client.post("/api/v1/distributed/compare-all", json={
            "num_gpus": 4,
            "model_params_m": 100,
            "batch_size": 32,
            "gpu_memory_gb": 24,
        })
        assert r.status_code == 200

    def test_strategies(self):
        r = client.get("/api/v1/distributed/strategies")
        assert r.status_code == 200


class TestPromptEngAPI:
    def test_templates(self):
        r = client.get("/api/v1/prompt-eng/templates")
        assert r.status_code == 200

    def test_analyze(self):
        r = client.post("/api/v1/prompt-eng/analyze", json={
            "text": "Let's think step by step. What is 2+2?",
        })
        assert r.status_code == 200

    def test_techniques(self):
        r = client.get("/api/v1/prompt-eng/techniques")
        assert r.status_code == 200


class TestSafetyAPI:
    def test_evaluate(self):
        r = client.post("/api/v1/safety/evaluate", json={
            "text": "This is a normal test sentence.",
        })
        assert r.status_code == 200

    def test_redteam(self):
        r = client.post("/api/v1/safety/redteam")
        assert r.status_code == 200

    def test_categories(self):
        r = client.get("/api/v1/safety/categories")
        assert r.status_code == 200

    def test_principles(self):
        r = client.get("/api/v1/safety/principles")
        assert r.status_code == 200


class TestLongContextAPI:
    def test_rope_frequencies(self):
        r = client.post("/api/v1/long-context/rope/frequencies", json={
            "dim": 64,
            "max_position": 128,
            "method": "none",
            "scaling_factor": 1,
        })
        assert r.status_code == 200

    def test_alibi_bias_matrix(self):
        r = client.post("/api/v1/long-context/alibi/bias-matrix", json={
            "num_heads": 4,
            "seq_len": 8,
        })
        assert r.status_code == 200

    def test_compare(self):
        r = client.post("/api/v1/long-context/compare", json={
            "dim": 64,
            "num_heads": 4,
            "max_position": 128,
        })
        assert r.status_code == 200
