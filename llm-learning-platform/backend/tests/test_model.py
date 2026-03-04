"""Comprehensive tests for MicroGPT model and GPTConfig."""

import pytest
import numpy as np
from app.core.model import MicroGPT, GPTConfig, PRESET_CONFIGS


class TestGPTConfig:
    def test_default_config(self):
        config = GPTConfig()
        assert config.vocab_size == 256
        assert config.d_model == 128
        assert config.num_heads == 4
        assert config.num_layers == 4

    def test_num_parameters(self):
        config = GPTConfig()
        num = config.num_parameters  # @property, not a method
        assert isinstance(num, int)
        assert num > 0

    def test_preset_configs_exist(self):
        assert "nano" in PRESET_CONFIGS
        assert "small" in PRESET_CONFIGS
        assert "medium" in PRESET_CONFIGS

    def test_nano_preset(self):
        config = PRESET_CONFIGS["nano"]
        assert config.d_model == 64
        assert config.num_heads == 2
        assert config.num_layers == 2


class TestMicroGPT:
    @pytest.fixture
    def nano_model(self):
        config = PRESET_CONFIGS["nano"]
        return MicroGPT(config)

    def test_init(self, nano_model):
        assert nano_model is not None
        assert nano_model.config.d_model == 64

    def test_forward_logits_shape(self, nano_model):
        batch_size, seq_len = 2, 8
        token_ids = np.random.randint(0, 256, size=(batch_size, seq_len))
        result = nano_model.forward(token_ids)
        assert "logits" in result
        logits = result["logits"]
        # Logits shape: (batch, seq_len, vocab_size)
        assert logits.shape[0] == batch_size
        assert logits.shape[1] == seq_len
        assert logits.shape[2] == nano_model.config.vocab_size

    def test_forward_with_targets(self, nano_model):
        batch_size, seq_len = 2, 8
        token_ids = np.random.randint(0, 256, size=(batch_size, seq_len))
        targets = np.random.randint(0, 256, size=(batch_size, seq_len))
        result = nano_model.forward(token_ids, targets=targets)
        assert "loss" in result
        assert isinstance(result["loss"], float) or isinstance(result["loss"], np.floating)
        assert result["loss"] > 0  # Cross-entropy loss should be positive

    def test_forward_with_intermediates(self, nano_model):
        token_ids = np.random.randint(0, 256, size=(1, 4))
        result = nano_model.forward(token_ids, store_intermediates=True)
        assert "intermediates" in result
        assert len(result["intermediates"]) > 0

    def test_generate(self, nano_model):
        prompt = np.array([[1, 2, 3, 4]])
        generated, metadata = nano_model.generate(
            prompt, max_new_tokens=5, temperature=1.0
        )
        # Generated should be longer than prompt
        assert generated.shape[1] > prompt.shape[1]
        assert generated.shape[1] <= prompt.shape[1] + 5

    def test_parameters(self, nano_model):
        params = list(nano_model.parameters())
        assert isinstance(params, list)
        assert len(params) > 0

    def test_num_parameters(self, nano_model):
        info = nano_model.parameters_count()
        assert isinstance(info, dict)
        assert info["total"] > 0
        assert info["trainable"] > 0

    def test_set_training(self, nano_model):
        nano_model.set_training(True)
        nano_model.set_training(False)  # Should not raise

    def test_zero_grad(self, nano_model):
        nano_model.zero_grad()  # Should not raise

    def test_config_summary(self, nano_model):
        info = nano_model.parameters_count()
        assert isinstance(info, dict)
        assert "total" in info
        assert "size_mb" in info

    def test_save_load_checkpoint(self, nano_model, tmp_path):
        path = str(tmp_path / "test_checkpoint.pt")
        nano_model.save_checkpoint(path)
        # Verify that loading doesn't error
        config = PRESET_CONFIGS["nano"]
        model2 = MicroGPT(config)
        model2.load_checkpoint(path)
        # Verify model2 can still produce output
        token_ids = np.array([[1, 2, 3]])
        r2 = model2.forward(token_ids)
        assert "logits" in r2
        assert r2["logits"].shape == (1, 3, config.vocab_size)
