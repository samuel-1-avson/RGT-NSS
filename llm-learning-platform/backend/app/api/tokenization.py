"""
Tokenization API

Interactive tokenization with multiple strategies,
training visualization, and comparison tools.
"""

from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.core.tokenizer import TokenizerEngine, TokenizationStrategy

router = APIRouter()


class TokenizeRequest(BaseModel):
    text: str = Field(..., max_length=10000)
    strategy: str = "character"
    add_special_tokens: bool = True


class TrainTokenizerRequest(BaseModel):
    corpus: str = Field(..., max_length=100000)
    strategy: str = "bpe"
    vocab_size: int = Field(default=500, ge=10, le=50000)


class CompareRequest(BaseModel):
    text: str = Field(..., max_length=10000)
    strategies: List[str] = ["character", "word", "bpe"]


@router.post("/tokenize")
async def tokenize_text(request: TokenizeRequest):
    """Tokenize text using the specified strategy."""
    strategy = TokenizationStrategy(request.strategy)
    engine = TokenizerEngine(strategy)

    # Quick train on the input text for BPE/WordPiece
    if strategy in (TokenizationStrategy.BPE, TokenizationStrategy.WORDPIECE):
        engine.train(request.text, vocab_size=500)

    result = engine.encode(request.text, add_special_tokens=request.add_special_tokens)
    stats = engine.get_stats()

    return {
        "tokens": result.tokens,
        "token_ids": result.ids,
        "token_count": len(result.tokens),
        "unknown_count": result.unknown_count,
        "vocab_size": stats.vocab_size,
        "compression_ratio": len(request.text) / max(len(result.tokens), 1),
        "strategy": request.strategy,
    }


@router.post("/train")
async def train_tokenizer(request: TrainTokenizerRequest):
    """Train a tokenizer on a corpus with step-by-step visualization."""
    strategy = TokenizationStrategy(request.strategy)
    engine = TokenizerEngine(strategy)

    if strategy == TokenizationStrategy.BPE:
        steps = list(engine.train_bpe_streaming(request.corpus, request.vocab_size))
        return {
            "strategy": request.strategy,
            "final_vocab_size": len(engine.vocab),
            "num_merges": len(steps),
            "merge_steps": [
                {
                    "step": s.step,
                    "pair": list(s.merge),
                    "new_token": s.new_token,
                    "frequency": s.frequency,
                    "vocab_size": s.vocab_size,
                    "compression_ratio": s.compression_ratio,
                }
                for s in steps
            ],
            "vocabulary": dict(
                sorted(engine.vocab.items(), key=lambda x: x[1])[:100]
            ),
        }
    else:
        result = engine.train(request.corpus, request.vocab_size)
        return {
            "strategy": request.strategy,
            "final_vocab_size": result.final_size,
            "compression_ratio": result.compression_ratio,
            "vocabulary": dict(
                sorted(engine.vocab.items(), key=lambda x: x[1])[:100]
            ),
        }


@router.post("/compare")
async def compare_strategies(request: CompareRequest):
    """Compare multiple tokenization strategies side-by-side."""
    strategies = [TokenizationStrategy(s) for s in request.strategies]
    engine = TokenizerEngine()
    results = engine.compare_strategies(request.text, strategies)

    # Frontend expects an array for rendering with .map(); keep dict form too.
    comparisons = [
        {
            "strategy": strategy,
            **payload,
        }
        for strategy, payload in results.items()
    ]

    return {
        "text": request.text,
        "comparisons": comparisons,
        "comparisons_by_strategy": results,
    }


@router.post("/frequencies")
async def token_frequencies(request: TokenizeRequest):
    """Get token frequency distribution."""
    strategy = TokenizationStrategy(request.strategy)
    engine = TokenizerEngine(strategy)
    if strategy in (TokenizationStrategy.BPE, TokenizationStrategy.WORDPIECE):
        engine.train(request.text, vocab_size=500)
    freqs = engine.get_token_frequencies(request.text)
    sorted_freqs = sorted(freqs.items(), key=lambda x: x[1], reverse=True)
    return {
        "frequencies": [{"token": t, "count": c} for t, c in sorted_freqs],
        "unique_tokens": len(freqs),
        "total_tokens": sum(freqs.values()),
    }
