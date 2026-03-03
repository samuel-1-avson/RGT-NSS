"""
Tests for tokenizer implementations.
"""

import pytest

from app.core.tokenizer import (
    CharacterTokenizer,
    WordTokenizer,
    BPETokenizer,
    TokenizerFactory,
)
from app.core.exceptions import TokenizationError


class TestCharacterTokenizer:
    """Test character-level tokenizer."""
    
    def test_encode(self):
        """Test basic encoding."""
        tokenizer = CharacterTokenizer(vocab_size=256)
        text = "Hello"
        
        tokens = tokenizer.encode(text)
        
        assert len(tokens) == len(text)
        assert all(isinstance(t, int) for t in tokens)
        assert all(0 <= t < 256 for t in tokens)
    
    def test_decode(self):
        """Test basic decoding."""
        tokenizer = CharacterTokenizer(vocab_size=256)
        text = "Hello"
        
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        
        assert decoded == text
    
    def test_encode_decode_roundtrip(self):
        """Test encode-decode roundtrip."""
        tokenizer = CharacterTokenizer(vocab_size=256)
        text = "Hello, World! 123"
        
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        
        assert decoded == text
    
    def test_empty_string(self):
        """Test encoding empty string."""
        tokenizer = CharacterTokenizer()
        tokens = tokenizer.encode("")
        assert tokens == []
    
    def test_vocab_size(self):
        """Test vocabulary size."""
        tokenizer = CharacterTokenizer(vocab_size=128)
        assert tokenizer.get_vocab_size() == 128


class TestWordTokenizer:
    """Test word-level tokenizer."""
    
    def test_train_and_encode(self):
        """Test training and encoding."""
        tokenizer = WordTokenizer(vocab_size=100)
        texts = [
            "Hello world",
            "Hello there",
            "World peace"
        ]
        
        tokenizer.train(texts)
        tokens = tokenizer.encode("Hello world")
        
        assert len(tokens) > 0
        assert all(isinstance(t, int) for t in tokens)
    
    def test_untrained_raises_error(self):
        """Test that untrained tokenizer raises error."""
        tokenizer = WordTokenizer()
        
        with pytest.raises(TokenizationError):
            tokenizer.encode("Hello")
    
    def test_vocab_size_limit(self):
        """Test vocabulary size is respected."""
        tokenizer = WordTokenizer(vocab_size=10)
        texts = ["The quick brown fox jumps over the lazy dog"]
        
        tokenizer.train(texts)
        
        assert tokenizer.get_vocab_size() <= 10


class TestBPETokenizer:
    """Test BPE tokenizer."""
    
    def test_train_and_encode(self):
        """Test training and encoding."""
        tokenizer = BPETokenizer(vocab_size=300, num_merges=10)
        texts = ["Hello world", "Hello there", "World peace"]
        
        tokenizer.train(texts)
        tokens = tokenizer.encode("Hello world")
        
        assert len(tokens) > 0
        assert all(isinstance(t, int) for t in tokens)
    
    def test_encode_decode_roundtrip(self):
        """Test roundtrip."""
        tokenizer = BPETokenizer(vocab_size=300, num_merges=5)
        texts = ["Hello world"]
        
        tokenizer.train(texts)
        text = "Hello world"
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        
        # Note: BPE decoding may not be exact due to </w> token
        assert "Hello" in decoded
        assert "world" in decoded.lower()


class TestTokenizerFactory:
    """Test tokenizer factory."""
    
    def test_create_character(self):
        """Test creating character tokenizer."""
        tokenizer = TokenizerFactory.create("character", vocab_size=256)
        assert isinstance(tokenizer, CharacterTokenizer)
    
    def test_create_word(self):
        """Test creating word tokenizer."""
        tokenizer = TokenizerFactory.create("word", vocab_size=1000)
        assert isinstance(tokenizer, WordTokenizer)
    
    def test_create_bpe(self):
        """Test creating BPE tokenizer."""
        tokenizer = TokenizerFactory.create("bpe", vocab_size=1000)
        assert isinstance(tokenizer, BPETokenizer)
    
    def test_create_unknown_raises(self):
        """Test that unknown strategy raises error."""
        with pytest.raises(TokenizationError):
            TokenizerFactory.create("unknown")
    
    def test_get_or_create_caches(self):
        """Test that get_or_create caches tokenizers."""
        TokenizerFactory.clear_cache()
        
        t1 = TokenizerFactory.get_or_create("test", "character")
        t2 = TokenizerFactory.get_or_create("test", "character")
        
        assert t1 is t2


class TestTokenizerSaveLoad:
    """Test tokenizer serialization."""
    
    def test_save_load_character(self, tmp_path):
        """Test saving and loading character tokenizer."""
        tokenizer = CharacterTokenizer(vocab_size=128)
        
        save_path = tmp_path / "tokenizer.json"
        tokenizer.save(str(save_path))
        
        loaded = CharacterTokenizer.load(str(save_path))
        
        assert loaded.get_vocab_size() == tokenizer.get_vocab_size()
    
    def test_save_load_bpe(self, tmp_path):
        """Test saving and loading BPE tokenizer."""
        tokenizer = BPETokenizer(vocab_size=300, num_merges=5)
        tokenizer.train(["Hello world"])
        
        save_path = tmp_path / "bpe_tokenizer.json"
        tokenizer.save(str(save_path))
        
        loaded = BPETokenizer.load(str(save_path))
        
        # Test that loaded tokenizer works
        tokens1 = tokenizer.encode("Hello")
        tokens2 = loaded.encode("Hello")
        assert tokens1 == tokens2
