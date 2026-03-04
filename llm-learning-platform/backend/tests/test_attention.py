"""Comprehensive tests for MultiHeadAttention with PyTorch backend."""

import pytest
import torch
import numpy as np
from app.core.attention import MultiHeadAttention, AttentionType


class TestMultiHeadAttention:
    @pytest.fixture
    def mha(self):
        return MultiHeadAttention(d_model=64, num_heads=4)

    def test_init(self, mha):
        assert mha is not None

    def test_forward_shape(self, mha):
        x = torch.randn(2, 8, 64)
        output = mha.forward(x)
        assert output.shape == (2, 8, 64)

    def test_forward_with_causal_mask(self, mha):
        seq_len = 8
        x = torch.randn(1, seq_len, 64)
        mask = MultiHeadAttention.create_causal_mask(seq_len)
        output = mha.forward(x, mask=mask)
        assert output.shape == (1, seq_len, 64)

    def test_step_by_step(self, mha):
        x = torch.randn(1, 4, 64)
        result = mha.forward_step_by_step(x)
        assert "attention_weights" in result
        assert "queries" in result

    def test_step_by_step_keys(self, mha):
        x = torch.randn(1, 4, 64)
        result = mha.forward_step_by_step(x)
        assert "queries" in result
        assert "keys" in result
        assert "values" in result

    def test_parameters(self, mha):
        params = list(mha.parameters())
        assert isinstance(params, list)
        assert len(params) > 0

    def test_store_weights(self, mha):
        x = torch.randn(1, 4, 64)
        mha.forward(x, store_weights=True)
        weights = mha.get_attention_weights()
        assert weights is not None
        assert weights.shape[0] == 1  # batch
        assert weights.shape[1] == 4  # heads


class TestCausalMask:
    def test_shape(self):
        mask = MultiHeadAttention.create_causal_mask(5)
        assert mask.shape == (5, 5)

    def test_values(self):
        mask = MultiHeadAttention.create_causal_mask(3)
        # Lower triangle should be 1
        assert mask[0, 0].item() == 1.0
        assert mask[2, 0].item() == 1.0
        # Upper triangle should be 0
        assert mask[0, 2].item() == 0.0


class TestAttentionVariants:
    def test_local_attention(self):
        mha = MultiHeadAttention(
            d_model=64, num_heads=4,
            attention_type=AttentionType.LOCAL,
            local_window=4,
        )
        x = torch.randn(1, 16, 64)
        output = mha.forward(x)
        assert output.shape == (1, 16, 64)

    def test_sparse_attention(self):
        mha = MultiHeadAttention(
            d_model=64, num_heads=4,
            attention_type=AttentionType.SPARSE,
        )
        x = torch.randn(1, 8, 64)
        output = mha.forward(x)
        assert output.shape == (1, 8, 64)

    def test_linear_attention(self):
        mha = MultiHeadAttention(
            d_model=64, num_heads=4,
            attention_type=AttentionType.LINEAR,
        )
        x = torch.randn(1, 8, 64)
        output = mha.forward(x)
        assert output.shape == (1, 8, 64)


class TestAttentionPatternAnalysis:
    def test_analyze(self):
        mha = MultiHeadAttention(d_model=64, num_heads=4)
        x = torch.randn(1, 6, 64)
        result = mha.analyze_attention_patterns(x)
        assert "head_patterns" in result
        assert len(result["head_patterns"]) > 0
