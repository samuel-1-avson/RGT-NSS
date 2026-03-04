"""
Tokenizer Engine

Comprehensive tokenization supporting character, word, BPE, WordPiece,
and SentencePiece strategies — all built from scratch for educational
transparency.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Generator, List, Optional, Tuple


class TokenizationStrategy(str, Enum):
    CHARACTER = "character"
    WORD = "word"
    WHITESPACE = "whitespace"
    BPE = "bpe"
    WORDPIECE = "wordpiece"
    UNIGRAM = "unigram"


# ─── Data Classes ────────────────────────────────────────────

@dataclass
class Token:
    text: str
    id: int
    start: int
    end: int
    is_special: bool = False


@dataclass
class BPEMerge:
    pair: Tuple[str, str]
    new_token: str
    frequency: int
    step: int


@dataclass
class BPEStepResult:
    step: int
    merge: Tuple[str, str]
    new_token: str
    frequency: int
    vocab_size: int
    compression_ratio: float


@dataclass
class BPETrainingResult:
    vocab: Dict[str, int]
    merges: List[BPEMerge]
    final_size: int
    compression_ratio: float
    training_steps: int


@dataclass
class EncodingResult:
    ids: List[int]
    tokens: List[str]
    metadata: List[dict]
    unknown_count: int


@dataclass
class TokenizerStats:
    vocab_size: int
    merge_rules_count: int
    special_tokens: Dict[str, int]
    strategy: str


# ─── Tokenizer Engine ───────────────────────────────────────

class TokenizerEngine:
    """
    Comprehensive tokenization engine supporting multiple strategies.
    Built from scratch for educational transparency.
    """

    SPECIAL_TOKENS = {
        "<|PAD|>": 0,
        "<|UNK|>": 1,
        "<|BOS|>": 2,
        "<|EOS|>": 3,
        "<|MASK|>": 4,
    }

    def __init__(self, strategy: TokenizationStrategy = TokenizationStrategy.CHARACTER):
        self.strategy = strategy
        self.vocab: Dict[str, int] = dict(self.SPECIAL_TOKENS)
        self.inverse_vocab: Dict[int, str] = {v: k for k, v in self.vocab.items()}
        self.merge_rules: List[BPEMerge] = []
        self._next_id = len(self.SPECIAL_TOKENS)

    # ─── Training ────────────────────────────────────────────

    def train(
        self,
        corpus: str,
        vocab_size: int = 1000,
    ) -> BPETrainingResult:
        """
        Train the tokenizer on a corpus.
        Dispatches to the appropriate strategy.
        """
        if self.strategy == TokenizationStrategy.BPE:
            return self._train_bpe(corpus, vocab_size)
        elif self.strategy == TokenizationStrategy.WORDPIECE:
            return self._train_wordpiece(corpus, vocab_size)
        elif self.strategy == TokenizationStrategy.CHARACTER:
            return self._train_character(corpus)
        elif self.strategy in (TokenizationStrategy.WORD, TokenizationStrategy.WHITESPACE):
            return self._train_word(corpus, vocab_size)
        elif self.strategy == TokenizationStrategy.UNIGRAM:
            return self._train_unigram(corpus, vocab_size)
        raise ValueError(f"Unknown strategy: {self.strategy}")

    def train_bpe_streaming(
        self,
        corpus: str,
        vocab_size: int = 1000,
    ) -> Generator[BPEStepResult, None, None]:
        """Train BPE with step-by-step results for visualization."""
        # Initialize character vocabulary
        chars = sorted(set(corpus))
        for ch in chars:
            self._add_token(ch)

        # Pre-tokenize into words
        words = self._pre_tokenize(corpus)
        word_freqs = Counter(words)

        # Convert words to character sequences
        splits: Dict[str, List[str]] = {word: list(word) for word in word_freqs}

        step = 0
        while len(self.vocab) < vocab_size:
            pair_freqs = self._get_pair_frequencies(splits, word_freqs)
            if not pair_freqs:
                break

            best_pair = max(pair_freqs, key=pair_freqs.get)
            merged_token = "".join(best_pair)

            splits = self._merge_pair(best_pair, splits)
            self._add_token(merged_token)

            merge = BPEMerge(
                pair=best_pair,
                new_token=merged_token,
                frequency=pair_freqs[best_pair],
                step=step,
            )
            self.merge_rules.append(merge)

            compression = self._compute_compression(corpus, splits, word_freqs)

            yield BPEStepResult(
                step=step,
                merge=best_pair,
                new_token=merged_token,
                frequency=pair_freqs[best_pair],
                vocab_size=len(self.vocab),
                compression_ratio=compression,
            )
            step += 1

    def _train_bpe(self, corpus: str, vocab_size: int) -> BPETrainingResult:
        """Train BPE tokenizer (non-streaming)."""
        steps = list(self.train_bpe_streaming(corpus, vocab_size))
        last_step = steps[-1] if steps else None
        return BPETrainingResult(
            vocab=dict(self.vocab),
            merges=list(self.merge_rules),
            final_size=len(self.vocab),
            compression_ratio=last_step.compression_ratio if last_step else 1.0,
            training_steps=len(steps),
        )

    def _train_wordpiece(self, corpus: str, vocab_size: int) -> BPETrainingResult:
        """Train WordPiece tokenizer (likelihood-based merging)."""
        chars = sorted(set(corpus))
        for ch in chars:
            self._add_token(ch)

        words = self._pre_tokenize(corpus)
        word_freqs = Counter(words)
        splits: Dict[str, List[str]] = {}
        for word in word_freqs:
            chars_list = list(word)
            # WordPiece uses ## prefix for continuation tokens
            splits[word] = [chars_list[0]] + ["##" + c for c in chars_list[1:]]
            for token in splits[word]:
                if token not in self.vocab:
                    self._add_token(token)

        step = 0
        while len(self.vocab) < vocab_size:
            pair_scores = self._get_wordpiece_scores(splits, word_freqs)
            if not pair_scores:
                break

            best_pair = max(pair_scores, key=pair_scores.get)
            merged = best_pair[0] + best_pair[1].lstrip("#")
            if best_pair[1].startswith("##"):
                merged = best_pair[0] + best_pair[1][2:]

            splits = self._merge_pair(best_pair, splits)
            self._add_token(merged)
            self.merge_rules.append(
                BPEMerge(pair=best_pair, new_token=merged, frequency=0, step=step)
            )
            step += 1

        return BPETrainingResult(
            vocab=dict(self.vocab),
            merges=list(self.merge_rules),
            final_size=len(self.vocab),
            compression_ratio=1.0,
            training_steps=step,
        )

    def _train_character(self, corpus: str) -> BPETrainingResult:
        """Build character-level vocabulary."""
        for ch in sorted(set(corpus)):
            self._add_token(ch)
        return BPETrainingResult(
            vocab=dict(self.vocab),
            merges=[],
            final_size=len(self.vocab),
            compression_ratio=1.0,
            training_steps=0,
        )

    def _train_word(self, corpus: str, vocab_size: int) -> BPETrainingResult:
        """Build word-level vocabulary (top-k by frequency)."""
        words = self._pre_tokenize(corpus)
        word_freqs = Counter(words)
        top_k = word_freqs.most_common(vocab_size - len(self.SPECIAL_TOKENS))
        for word, _ in top_k:
            self._add_token(word)
        return BPETrainingResult(
            vocab=dict(self.vocab),
            merges=[],
            final_size=len(self.vocab),
            compression_ratio=1.0,
            training_steps=0,
        )

    def _train_unigram(self, corpus: str, vocab_size: int) -> BPETrainingResult:
        """Train Unigram (SentencePiece-style) tokenizer via EM pruning."""
        import math

        # Build initial oversized vocabulary from character n-grams
        char_counter: Counter = Counter()
        for n in range(1, 6):
            for i in range(len(corpus) - n + 1):
                char_counter[corpus[i : i + n]] += 1

        # Keep top 5× target vocab_size as seed vocabulary
        seed_size = min(len(char_counter), vocab_size * 5)
        seed_tokens = [t for t, _ in char_counter.most_common(seed_size)]
        for t in seed_tokens:
            self._add_token(t)

        # Assign initial log-probabilities proportional to frequency
        total = sum(char_counter[t] for t in seed_tokens)
        log_probs: Dict[str, float] = {}
        for t in seed_tokens:
            log_probs[t] = math.log(char_counter[t] / total) if char_counter[t] > 0 else -20.0

        # EM pruning iterations
        steps = 0
        while len(log_probs) > vocab_size:
            # E-step: Viterbi segment the corpus to estimate token usage
            usage: Counter = Counter()
            pos = 0
            while pos < len(corpus):
                best_token = corpus[pos]
                best_score = log_probs.get(corpus[pos], -100.0)
                for end in range(min(pos + 5, len(corpus)), pos, -1):
                    cand = corpus[pos:end]
                    if cand in log_probs and log_probs[cand] > best_score:
                        best_token = cand
                        best_score = log_probs[cand]
                usage[best_token] += 1
                pos += len(best_token)

            # M-step: recompute probabilities from usage
            total_usage = sum(usage.values())
            for t in list(log_probs.keys()):
                log_probs[t] = math.log(usage.get(t, 0.5) / total_usage)

            # Prune: remove lowest-score tokens (keep at least vocab_size)
            prune_count = max((len(log_probs) - vocab_size) // 4, 1)
            sorted_tokens = sorted(log_probs.items(), key=lambda x: x[1])
            # Never prune single characters
            to_remove = [t for t, _ in sorted_tokens if len(t) > 1][:prune_count]
            for t in to_remove:
                del log_probs[t]
            steps += 1

        # Rebuild vocab from surviving tokens
        self.vocab = dict(self.SPECIAL_TOKENS)
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self._next_id = len(self.SPECIAL_TOKENS)
        for t in log_probs:
            self._add_token(t)

        return BPETrainingResult(
            vocab=dict(self.vocab),
            merges=[],
            final_size=len(self.vocab),
            compression_ratio=1.0,
            training_steps=steps,
        )

    # ─── Encoding ────────────────────────────────────────────

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
    ) -> EncodingResult:
        """Encode text to token IDs."""
        if self.strategy == TokenizationStrategy.CHARACTER:
            tokens = list(text)
        elif self.strategy in (TokenizationStrategy.WORD, TokenizationStrategy.WHITESPACE):
            tokens = self._pre_tokenize(text)
        elif self.strategy == TokenizationStrategy.BPE:
            tokens = self._encode_bpe(text)
        elif self.strategy == TokenizationStrategy.WORDPIECE:
            tokens = self._encode_wordpiece(text)
        elif self.strategy == TokenizationStrategy.UNIGRAM:
            tokens = self._encode_unigram(text)
        else:
            tokens = list(text)

        unk_id = self.SPECIAL_TOKENS["<|UNK|>"]
        token_ids = [self.vocab.get(t, unk_id) for t in tokens]
        unknown_count = sum(1 for t in tokens if t not in self.vocab)

        metadata = [
            {"token": t, "id": self.vocab.get(t, unk_id), "is_unk": t not in self.vocab}
            for t in tokens
        ]

        if add_special_tokens:
            bos = self.SPECIAL_TOKENS["<|BOS|>"]
            eos = self.SPECIAL_TOKENS["<|EOS|>"]
            token_ids = [bos] + token_ids + [eos]
            tokens = ["<|BOS|>"] + tokens + ["<|EOS|>"]

        return EncodingResult(
            ids=token_ids,
            tokens=tokens,
            metadata=metadata,
            unknown_count=unknown_count,
        )

    def _encode_bpe(self, text: str) -> List[str]:
        """Apply BPE merges to encode text."""
        words = self._pre_tokenize(text)
        all_tokens: List[str] = []
        for word in words:
            chars = list(word)
            for merge in self.merge_rules:
                chars = self._apply_merge(chars, merge.pair)
            all_tokens.extend(chars)
        return all_tokens

    def _encode_wordpiece(self, text: str) -> List[str]:
        """Encode using greedy longest-match WordPiece."""
        words = self._pre_tokenize(text)
        all_tokens: List[str] = []
        for word in words:
            i = 0
            sub_tokens: List[str] = []
            while i < len(word):
                best_match = None
                for end in range(len(word), i, -1):
                    candidate = word[i:end]
                    if i > 0:
                        candidate = "##" + candidate
                    if candidate in self.vocab:
                        best_match = candidate
                        i = end
                        break
                if best_match:
                    sub_tokens.append(best_match)
                else:
                    sub_tokens.append("<|UNK|>")
                    i += 1
            all_tokens.extend(sub_tokens)
        return all_tokens

    def _encode_unigram(self, text: str) -> List[str]:
        """Greedy longest-match Unigram encoding."""
        tokens: List[str] = []
        pos = 0
        while pos < len(text):
            best_token = text[pos]
            best_len = 1
            for end in range(min(pos + 20, len(text)), pos, -1):
                cand = text[pos:end]
                if cand in self.vocab and (end - pos) > best_len:
                    best_token = cand
                    best_len = end - pos
            tokens.append(best_token)
            pos += best_len
        return tokens

    # ─── Decoding ────────────────────────────────────────────

    def decode(self, token_ids: List[int], skip_special: bool = True) -> str:
        """Decode token IDs back to text."""
        special_ids = set(self.SPECIAL_TOKENS.values()) if skip_special else set()
        tokens = []
        for tid in token_ids:
            if tid in special_ids:
                continue
            token = self.inverse_vocab.get(tid, "")
            if token.startswith("##"):
                token = token[2:]
            tokens.append(token)
        return "".join(tokens)

    # ─── Analysis / Visualization ────────────────────────────

    def compare_strategies(
        self, text: str, strategies: List[TokenizationStrategy]
    ) -> Dict[str, dict]:
        """Compare multiple tokenization strategies on the same text."""
        results = {}
        for strategy in strategies:
            engine = TokenizerEngine(strategy)
            engine.train(text, vocab_size=500)
            enc = engine.encode(text, add_special_tokens=False)
            results[strategy.value] = {
                "tokens": enc.tokens,
                "token_count": len(enc.tokens),
                "vocab_size": len(engine.vocab),
                "compression_ratio": len(text) / max(len(enc.tokens), 1),
                "unknown_count": enc.unknown_count,
            }
        return results

    def get_token_frequencies(self, text: str) -> Dict[str, int]:
        """Get token frequency distribution for encoded text."""
        enc = self.encode(text, add_special_tokens=False)
        return dict(Counter(enc.tokens))

    def get_stats(self) -> TokenizerStats:
        return TokenizerStats(
            vocab_size=len(self.vocab),
            merge_rules_count=len(self.merge_rules),
            special_tokens=dict(self.SPECIAL_TOKENS),
            strategy=self.strategy.value,
        )

    # ─── Serialization ───────────────────────────────────────

    def save(self, path: str):
        """Save tokenizer to JSON."""
        data = {
            "strategy": self.strategy.value,
            "vocab": self.vocab,
            "merges": [
                {
                    "pair": list(m.pair),
                    "new_token": m.new_token,
                    "frequency": m.frequency,
                    "step": m.step,
                }
                for m in self.merge_rules
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "TokenizerEngine":
        """Load tokenizer from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        engine = cls(TokenizationStrategy(data["strategy"]))
        engine.vocab = data["vocab"]
        engine.inverse_vocab = {v: k for k, v in engine.vocab.items()}
        engine._next_id = max(engine.vocab.values()) + 1
        engine.merge_rules = [
            BPEMerge(
                pair=tuple(m["pair"]),
                new_token=m["new_token"],
                frequency=m["frequency"],
                step=m["step"],
            )
            for m in data["merges"]
        ]
        return engine

    # ─── Internal Helpers ────────────────────────────────────

    def _add_token(self, token: str):
        if token not in self.vocab:
            self.vocab[token] = self._next_id
            self.inverse_vocab[self._next_id] = token
            self._next_id += 1

    def _pre_tokenize(self, text: str) -> List[str]:
        """Split text into words (basic pre-tokenization)."""
        return re.findall(r"\S+", text)

    def _get_pair_frequencies(
        self,
        splits: Dict[str, List[str]],
        word_freqs: Counter,
    ) -> Dict[Tuple[str, str], int]:
        pair_freqs: Dict[Tuple[str, str], int] = defaultdict(int)
        for word, freq in word_freqs.items():
            split = splits[word]
            for i in range(len(split) - 1):
                pair_freqs[(split[i], split[i + 1])] += freq
        return dict(pair_freqs)

    def _get_wordpiece_scores(
        self,
        splits: Dict[str, List[str]],
        word_freqs: Counter,
    ) -> Dict[Tuple[str, str], float]:
        pair_freqs: Dict[Tuple[str, str], int] = defaultdict(int)
        token_freqs: Dict[str, int] = defaultdict(int)
        for word, freq in word_freqs.items():
            split = splits[word]
            for i, tok in enumerate(split):
                token_freqs[tok] += freq
                if i < len(split) - 1:
                    pair_freqs[(tok, split[i + 1])] += freq
        scores = {}
        for pair, freq in pair_freqs.items():
            denom = token_freqs[pair[0]] * token_freqs[pair[1]]
            scores[pair] = freq / denom if denom > 0 else 0
        return scores

    def _merge_pair(
        self,
        pair: Tuple[str, str],
        splits: Dict[str, List[str]],
    ) -> Dict[str, List[str]]:
        new_splits = {}
        for word, split in splits.items():
            new_split: List[str] = []
            i = 0
            while i < len(split):
                if i < len(split) - 1 and (split[i], split[i + 1]) == pair:
                    merged = split[i] + split[i + 1].lstrip("#") if split[i + 1].startswith("##") else "".join(pair)
                    new_split.append(merged)
                    i += 2
                else:
                    new_split.append(split[i])
                    i += 1
            new_splits[word] = new_split
        return new_splits

    def _apply_merge(self, tokens: List[str], pair: Tuple[str, str]) -> List[str]:
        merged: List[str] = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                merged.append(tokens[i] + tokens[i + 1])
                i += 2
            else:
                merged.append(tokens[i])
                i += 1
        return merged

    def _compute_compression(
        self,
        corpus: str,
        splits: Dict[str, List[str]],
        word_freqs: Counter,
    ) -> float:
        total_chars = len(corpus)
        total_tokens = sum(
            len(splits[word]) * freq for word, freq in word_freqs.items()
        )
        return total_chars / max(total_tokens, 1)
