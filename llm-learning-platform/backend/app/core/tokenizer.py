"""
Tokenizer implementations for the LLM Learning Platform.
Supports multiple tokenization strategies including BPE and tiktoken.
"""

import re
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict
import json
import logging

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

from app.core.exceptions import TokenizationError


logger = logging.getLogger(__name__)


class BaseTokenizer:
    """Base class for all tokenizers."""
    
    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        raise NotImplementedError
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        raise NotImplementedError
    
    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """Encode multiple texts."""
        return [self.encode(text) for text in texts]
    
    def decode_batch(self, batch_ids: List[List[int]]) -> List[str]:
        """Decode multiple token ID sequences."""
        return [self.decode(ids) for ids in batch_ids]
    
    def get_vocab(self) -> Dict[str, int]:
        """Get vocabulary mapping."""
        return self.vocab.copy()
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.vocab)
    
    def save(self, path: str) -> None:
        """Save tokenizer to file."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'vocab': self.vocab,
                'vocab_size': self.vocab_size,
                'type': self.__class__.__name__
            }, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'BaseTokenizer':
        """Load tokenizer from file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tokenizer = cls(vocab_size=data['vocab_size'])
        tokenizer.vocab = data['vocab']
        tokenizer.inverse_vocab = {v: k for k, v in tokenizer.vocab.items()}
        return tokenizer


class CharacterTokenizer(BaseTokenizer):
    """Simple character-level tokenizer."""
    
    def __init__(self, vocab_size: int = 256):
        super().__init__(vocab_size)
        # Initialize with ASCII characters
        for i in range(min(vocab_size, 256)):
            char = chr(i)
            self.vocab[char] = i
            self.inverse_vocab[i] = char
    
    def encode(self, text: str) -> List[int]:
        """Encode text to character IDs."""
        return [ord(c) % self.vocab_size for c in text]
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode character IDs to text."""
        return ''.join(chr(i % 256) for i in token_ids)


class WordTokenizer(BaseTokenizer):
    """Word-level tokenizer."""
    
    def __init__(self, vocab_size: int = 10000):
        super().__init__(vocab_size)
        self.word_regex = re.compile(r'\b\w+\b|[^\w\s]')
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3,
        }
        self.vocab = self.special_tokens.copy()
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self._is_trained = False
    
    def train(self, texts: List[str]) -> None:
        """Train tokenizer on corpus."""
        word_counts = defaultdict(int)
        
        for text in texts:
            words = self.word_regex.findall(text.lower())
            for word in words:
                word_counts[word] += 1
        
        # Take most frequent words
        most_common = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Reserve space for special tokens
        start_idx = len(self.special_tokens)
        available = self.vocab_size - start_idx
        
        for word, count in most_common[:available]:
            if word not in self.vocab:
                idx = len(self.vocab)
                self.vocab[word] = idx
                self.inverse_vocab[idx] = word
        
        self._is_trained = True
        logger.info(f"Trained WordTokenizer with vocab size {len(self.vocab)}")
    
    def encode(self, text: str) -> List[int]:
        """Encode text to word IDs."""
        if not self._is_trained:
            raise TokenizationError("Tokenizer must be trained before encoding")
        
        words = self.word_regex.findall(text.lower())
        return [self.vocab.get(word, self.special_tokens['<UNK>']) for word in words]
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode word IDs to text."""
        words = []
        for idx in token_ids:
            word = self.inverse_vocab.get(idx, '<UNK>')
            if word not in self.special_tokens:
                words.append(word)
        return ' '.join(words)


class BPETokenizer(BaseTokenizer):
    """Byte-Pair Encoding tokenizer."""
    
    def __init__(self, vocab_size: int = 10000, num_merges: Optional[int] = None):
        super().__init__(vocab_size)
        self.num_merges = num_merges or (vocab_size - 256)
        self.merges: List[Tuple[str, str]] = []
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3,
        }
        self._is_trained = False
    
    def train(self, texts: List[str]) -> None:
        """Train BPE tokenizer on corpus."""
        # Initialize with character vocabulary
        char_vocab = set()
        for text in texts:
            char_vocab.update(text)
        
        # Start with special tokens + characters
        self.vocab = self.special_tokens.copy()
        for i, char in enumerate(sorted(char_vocab), start=len(self.special_tokens)):
            self.vocab[char] = i
        
        # Build word frequencies
        word_freqs = defaultdict(int)
        for text in texts:
            # Split into words (simplified)
            words = text.split()
            for word in words:
                # End-of-word symbol
                word_chars = tuple(word) + ('</w>',)
                word_freqs[word_chars] += 1
        
        # BPE merges
        for i in range(self.num_merges):
            if len(self.vocab) >= self.vocab_size:
                break
            
            # Count pairs
            pairs = defaultdict(int)
            for word, freq in word_freqs.items():
                for j in range(len(word) - 1):
                    pairs[(word[j], word[j + 1])] += freq
            
            if not pairs:
                break
            
            # Find most frequent pair
            best_pair = max(pairs, key=pairs.get)
            new_token = best_pair[0] + best_pair[1]
            
            if new_token in self.vocab:
                continue
            
            # Add to vocabulary
            self.vocab[new_token] = len(self.vocab)
            self.merges.append(best_pair)
            
            # Apply merge to all words
            new_word_freqs = {}
            for word, freq in word_freqs.items():
                new_word = self._apply_merge(word, best_pair)
                new_word_freqs[new_word] = freq
            word_freqs = new_word_freqs
        
        # Build inverse vocab
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self._is_trained = True
        
        logger.info(f"Trained BPETokenizer with vocab size {len(self.vocab)}")
    
    def _apply_merge(self, word: Tuple[str, ...], pair: Tuple[str, str]) -> Tuple[str, ...]:
        """Apply a merge to a word."""
        result = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                result.append(pair[0] + pair[1])
                i += 2
            else:
                result.append(word[i])
                i += 1
        return tuple(result)
    
    def encode(self, text: str) -> List[int]:
        """Encode text using BPE."""
        if not self._is_trained:
            raise TokenizationError("Tokenizer must be trained before encoding")
        
        tokens = []
        words = text.split()
        
        for word in words:
            word_tokens = list(word) + ['</w>']
            
            # Apply merges
            for pair in self.merges:
                new_tokens = []
                i = 0
                while i < len(word_tokens):
                    if i < len(word_tokens) - 1 and word_tokens[i] == pair[0] and word_tokens[i + 1] == pair[1]:
                        new_tokens.append(pair[0] + pair[1])
                        i += 2
                    else:
                        new_tokens.append(word_tokens[i])
                        i += 1
                word_tokens = new_tokens
            
            # Convert to IDs
            for token in word_tokens:
                token_id = self.vocab.get(token, self.special_tokens['<UNK>'])
                tokens.append(token_id)
        
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode BPE tokens."""
        tokens = [self.inverse_vocab.get(i, '<UNK>') for i in token_ids]
        text = ''.join(tokens)
        text = text.replace('</w>', ' ')
        return text.strip()
    
    def save(self, path: str) -> None:
        """Save BPE tokenizer."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'vocab': self.vocab,
                'merges': self.merges,
                'vocab_size': self.vocab_size,
                'type': 'BPETokenizer'
            }, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'BPETokenizer':
        """Load BPE tokenizer."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tokenizer = cls(vocab_size=data['vocab_size'])
        tokenizer.vocab = {k: int(v) for k, v in data['vocab'].items()}
        tokenizer.merges = [tuple(m) for m in data['merges']]
        tokenizer.inverse_vocab = {v: k for k, v in tokenizer.vocab.items()}
        tokenizer._is_trained = True
        return tokenizer


class TiktokenTokenizer(BaseTokenizer):
    """Wrapper for OpenAI's tiktoken tokenizer."""
    
    def __init__(self, encoding_name: str = "cl100k_base"):
        if not TIKTOKEN_AVAILABLE:
            raise ImportError("tiktoken not installed. Install with: pip install tiktoken")
        
        super().__init__()
        self.encoding_name = encoding_name
        self.encoding = tiktoken.get_encoding(encoding_name)
        self.vocab_size = self.encoding.n_vocab
        self._build_vocab()
    
    def _build_vocab(self) -> None:
        """Build vocabulary from tiktoken encoding."""
        # Note: tiktoken doesn't expose full vocab, so we use special handling
        self.vocab = {
            '<|endoftext|>': self.encoding.eot_token,
        }
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
    
    def encode(self, text: str, allowed_special: Optional[Set[str]] = None) -> List[int]:
        """Encode text using tiktoken."""
        if allowed_special is None:
            allowed_special = set()
        return self.encoding.encode(text, allowed_special=allowed_special)
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode tiktoken IDs."""
        return self.encoding.decode(token_ids)
    
    def encode_batch(self, texts: List[str], num_threads: int = 8) -> List[List[int]]:
        """Encode multiple texts efficiently."""
        return self.encoding.encode_batch(texts, num_threads=num_threads)
    
    def decode_batch(self, batch_ids: List[List[int]]) -> List[str]:
        """Decode multiple sequences."""
        return [self.decode(ids) for ids in batch_ids]


class TokenizerFactory:
    """Factory for creating tokenizers."""
    
    _tokenizers: Dict[str, BaseTokenizer] = {}
    
    @classmethod
    def create(cls, strategy: str = "character", **kwargs) -> BaseTokenizer:
        """Create a tokenizer."""
        strategy = strategy.lower()
        
        if strategy == "character":
            return CharacterTokenizer(**kwargs)
        elif strategy == "word":
            return WordTokenizer(**kwargs)
        elif strategy == "bpe":
            return BPETokenizer(**kwargs)
        elif strategy in ["gpt2", "cl100k_base", "p50k_base", "r50k_base"]:
            if not TIKTOKEN_AVAILABLE:
                raise ImportError("tiktoken not installed")
            return TiktokenTokenizer(encoding_name=strategy)
        else:
            raise TokenizationError(f"Unknown tokenization strategy: {strategy}")
    
    @classmethod
    def get_or_create(cls, name: str, strategy: str = "character", **kwargs) -> BaseTokenizer:
        """Get cached tokenizer or create new one."""
        if name not in cls._tokenizers:
            cls._tokenizers[name] = cls.create(strategy, **kwargs)
        return cls._tokenizers[name]
    
    @classmethod
    def clear_cache(cls) -> None:
        """Clear tokenizer cache."""
        cls._tokenizers.clear()


# Global default tokenizer
_default_tokenizer: Optional[BaseTokenizer] = None


def get_default_tokenizer() -> BaseTokenizer:
    """Get default tokenizer instance."""
    global _default_tokenizer
    if _default_tokenizer is None:
        _default_tokenizer = CharacterTokenizer()
    return _default_tokenizer


def set_default_tokenizer(tokenizer: BaseTokenizer) -> None:
    """Set default tokenizer."""
    global _default_tokenizer
    _default_tokenizer = tokenizer
