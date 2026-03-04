"""Tests for the Tokenizer Engine."""

import pytest
from app.core.tokenizer import TokenizerEngine, TokenizationStrategy


class TestTokenizerBPE:
    def test_train_bpe(self, sample_corpus):
        engine = TokenizerEngine(TokenizationStrategy.BPE)
        result = engine.train(sample_corpus, vocab_size=100)
        assert result.final_size > 0
        assert result.final_size <= 100
        assert len(result.vocab) > 0

    def test_encode_bpe(self, sample_corpus):
        engine = TokenizerEngine(TokenizationStrategy.BPE)
        engine.train(sample_corpus, vocab_size=100)
        enc = engine.encode("The quick brown fox")
        assert len(enc.ids) > 0
        assert len(enc.tokens) > 0
        assert enc.tokens[0] == "<|BOS|>"
        assert enc.tokens[-1] == "<|EOS|>"

    def test_decode_bpe(self, sample_corpus):
        engine = TokenizerEngine(TokenizationStrategy.BPE)
        engine.train(sample_corpus, vocab_size=100)
        enc = engine.encode("fox", add_special_tokens=False)
        decoded = engine.decode(enc.ids)
        assert "fox" in decoded

    def test_bpe_streaming(self, sample_corpus):
        engine = TokenizerEngine(TokenizationStrategy.BPE)
        steps = list(engine.train_bpe_streaming(sample_corpus, vocab_size=50))
        assert len(steps) > 0
        for step in steps:
            assert step.vocab_size > 0
            assert step.new_token is not None


class TestTokenizerWordPiece:
    def test_train_wordpiece(self, sample_corpus):
        engine = TokenizerEngine(TokenizationStrategy.WORDPIECE)
        result = engine.train(sample_corpus, vocab_size=100)
        assert result.final_size > 0

    def test_encode_wordpiece(self, sample_corpus):
        engine = TokenizerEngine(TokenizationStrategy.WORDPIECE)
        engine.train(sample_corpus, vocab_size=100)
        enc = engine.encode("The fox")
        assert len(enc.ids) > 0


class TestTokenizerCharacter:
    def test_train_character(self, sample_text):
        engine = TokenizerEngine(TokenizationStrategy.CHARACTER)
        result = engine.train(sample_text)
        assert result.final_size > 0

    def test_encode_character(self, sample_text):
        engine = TokenizerEngine(TokenizationStrategy.CHARACTER)
        engine.train(sample_text)
        enc = engine.encode("abc", add_special_tokens=False)
        assert len(enc.tokens) == 3


class TestTokenizerUnigram:
    def test_train_unigram(self, sample_corpus):
        engine = TokenizerEngine(TokenizationStrategy.UNIGRAM)
        result = engine.train(sample_corpus, vocab_size=50)
        assert result.final_size > 0
        assert result.final_size <= 60  # allow some margin

    def test_encode_unigram(self, sample_corpus):
        engine = TokenizerEngine(TokenizationStrategy.UNIGRAM)
        engine.train(sample_corpus, vocab_size=50)
        enc = engine.encode("The fox", add_special_tokens=False)
        assert len(enc.tokens) > 0


class TestTokenizerComparison:
    def test_compare_strategies(self, sample_text):
        engine = TokenizerEngine(TokenizationStrategy.BPE)
        results = engine.compare_strategies(
            sample_text,
            [TokenizationStrategy.CHARACTER, TokenizationStrategy.BPE],
        )
        assert "character" in results
        assert "bpe" in results
        assert results["character"]["token_count"] > 0

    def test_token_frequencies(self, sample_corpus):
        engine = TokenizerEngine(TokenizationStrategy.CHARACTER)
        engine.train(sample_corpus)
        freqs = engine.get_token_frequencies(sample_corpus)
        assert len(freqs) > 0

    def test_stats(self):
        engine = TokenizerEngine(TokenizationStrategy.BPE)
        stats = engine.get_stats()
        assert stats.strategy == "bpe"
