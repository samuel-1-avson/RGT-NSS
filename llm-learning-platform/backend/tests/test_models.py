"""
Tests for model implementations.
"""

import pytest
import numpy as np
import torch

from app.models.gpt import GPTConfig, MicroGPT
from app.core.tensor import Tensor


class TestGPTConfig:
    """Test GPTConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = GPTConfig()
        assert config.vocab_size == 256
        assert config.max_seq_len == 256
        assert config.d_model == 128
        assert config.num_layers == 4
        assert config.num_heads == 4
    
    def test_d_model_divisible_by_heads(self):
        """Test that d_model must be divisible by num_heads."""
        config = GPTConfig(d_model=128, num_heads=8)
        assert config.d_model % config.num_heads == 0
        assert config.d_head == 16
    
    def test_post_init_assertion(self):
        """Test that invalid config raises assertion."""
        with pytest.raises(AssertionError):
            GPTConfig(d_model=100, num_heads=7)  # Not divisible


class TestMicroGPT:
    """Test MicroGPT model."""
    
    def test_model_creation(self, gpt_config):
        """Test model can be created."""
        model = MicroGPT(gpt_config)
        assert model is not None
        assert model.config == gpt_config
    
    def test_forward_pass(self, gpt_config):
        """Test forward pass."""
        model = MicroGPT(gpt_config)
        batch_size = 2
        seq_len = 10
        
        input_ids = np.random.randint(0, gpt_config.vocab_size, (batch_size, seq_len))
        logits, loss, attention = model.forward(input_ids)
        
        assert isinstance(logits, Tensor)
        assert logits.shape == (batch_size, seq_len, gpt_config.vocab_size)
        assert loss is None  # No targets provided
        assert attention is None  # return_attention=False
    
    def test_forward_with_targets(self, gpt_config):
        """Test forward pass with targets."""
        model = MicroGPT(gpt_config)
        batch_size = 2
        seq_len = 10
        
        input_ids = np.random.randint(0, gpt_config.vocab_size, (batch_size, seq_len))
        targets = np.random.randint(0, gpt_config.vocab_size, (batch_size, seq_len))
        
        logits, loss, _ = model.forward(input_ids, targets=targets)
        
        assert isinstance(loss, Tensor)
        assert loss.shape == ()  # Scalar loss
        assert loss.data > 0  # Loss should be positive
    
    def test_forward_with_attention(self, gpt_config):
        """Test forward pass returning attention."""
        model = MicroGPT(gpt_config)
        input_ids = np.random.randint(0, gpt_config.vocab_size, (1, 5))
        
        logits, loss, attention = model.forward(input_ids, return_attention=True)
        
        assert attention is not None
        assert len(attention) == gpt_config.num_layers
    
    def test_generate(self, gpt_config):
        """Test text generation."""
        model = MicroGPT(gpt_config)
        model.eval()
        
        input_ids = np.array([[1, 2, 3]])
        max_new_tokens = 5
        
        output = model.generate(input_ids, max_new_tokens=max_new_tokens)
        
        assert output.shape[1] == input_ids.shape[1] + max_new_tokens
    
    def test_count_parameters(self, gpt_config):
        """Test parameter counting."""
        model = MicroGPT(gpt_config)
        count = model.count_parameters()
        
        assert count > 0
        # Rough check: should be in the millions for reasonable configs
        assert count > 100000
    
    def test_state_dict(self, gpt_config):
        """Test state dict save/load."""
        model = MicroGPT(gpt_config)
        state = model.state_dict()
        
        assert isinstance(state, dict)
        assert len(state) > 0
        
        # Check that all parameters are in state dict
        for name, param in model.named_parameters().items():
            assert name in state or any(name.startswith(m) for m in ['blocks'])
    
    def test_train_eval_modes(self, gpt_config):
        """Test train/eval mode switching."""
        model = MicroGPT(gpt_config)
        
        model.train()
        assert model.training == True
        
        model.eval()
        assert model.training == False


class TestPyTorchGPT:
    """Test PyTorch GPT model (if available)."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_pytorch_model_creation(self, pytorch_gpt_config):
        """Test PyTorch model can be created."""
        from app.models.pytorch_gpt import PyTorchGPT
        
        model = PyTorchGPT(pytorch_gpt_config)
        assert model is not None
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_pytorch_forward(self, pytorch_gpt_config):
        """Test PyTorch forward pass."""
        from app.models.pytorch_gpt import PyTorchGPT
        
        model = PyTorchGPT(pytorch_gpt_config)
        batch_size = 2
        seq_len = 10
        
        input_ids = torch.randint(0, pytorch_gpt_config.vocab_size, (batch_size, seq_len))
        result = model.forward(input_ids)
        
        assert 'logits' in result
        assert result['logits'].shape == (batch_size, seq_len, pytorch_gpt_config.vocab_size)


class TestModelSerialization:
    """Test model serialization."""
    
    def test_save_load_state_dict(self, gpt_config, tmp_path):
        """Test saving and loading state dict."""
        model1 = MicroGPT(gpt_config)
        state1 = model1.state_dict()
        
        # Save
        save_path = tmp_path / "model_state.pkl"
        import pickle
        with open(save_path, 'wb') as f:
            pickle.dump(state1, f)
        
        # Load into new model
        model2 = MicroGPT(gpt_config)
        with open(save_path, 'rb') as f:
            state2 = pickle.load(f)
        
        model2.load_state_dict(state2)
        
        # Check parameters match
        for (n1, p1), (n2, p2) in zip(
            model1.named_parameters().items(),
            model2.named_parameters().items()
        ):
            assert n1 == n2
            np.testing.assert_array_equal(p1.data, p2.data)
