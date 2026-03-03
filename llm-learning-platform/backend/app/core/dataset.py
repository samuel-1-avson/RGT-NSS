"""
Dataset loading and processing for training.
"""

import numpy as np
import os
import re
from typing import List, Tuple, Iterator, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DatasetConfig:
    """Configuration for dataset loading."""
    name: str = "default"
    seq_length: int = 256
    vocab_size: int = 256
    train_split: float = 0.9
    shuffle: bool = True


class TextDataset:
    """
    Text dataset for language modeling.
    
    Supports various text sources and handles:
    - Tokenization
    - Batching
    - Train/validation splitting
    - Shuffling
    """
    
    # Sample datasets for quick testing
    SAMPLE_DATASETS = {
        'shakespeare': """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die—to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream—ay, there's the rub:
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause—there's the respect
That makes calamity of so long life.
""",
        'alice': """
Alice was beginning to get very tired of sitting by her sister on the bank,
and of having nothing to do: once or twice she had peeped into the book her
sister was reading, but it had no pictures or conversations in it, "and what
is the use of a book," thought Alice, "without pictures or conversations?"

So she was considering in her own mind (as well as she could, for the hot day
made her feel very sleepy and stupid), whether the pleasure of making a
daisy-chain would be worth the trouble of getting up and picking the daisies,
when suddenly a White Rabbit with pink eyes ran close by her.
""",
        'code': """
def hello_world():
    print("Hello, World!")
    return 42

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.weights = [np.random.randn(x, y) for x, y in zip(layers[:-1], layers[1:])]
    
    def forward(self, x):
        for w in self.weights:
            x = sigmoid(np.dot(x, w))
        return x
    
    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Backward pass
            error = y - output
            
            # Update weights
            for i in range(len(self.weights)):
                self.weights[i] += learning_rate * error
""",
    }
    
    def __init__(self, config: DatasetConfig = None):
        self.config = config or DatasetConfig()
        self.data: np.ndarray = np.array([], dtype=np.int32)
        self.train_data: np.ndarray = np.array([], dtype=np.int32)
        self.val_data: np.ndarray = np.array([], dtype=np.int32)
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        
    def load_sample(self, name: str) -> 'TextDataset':
        """Load a sample dataset."""
        if name not in self.SAMPLE_DATASETS:
            raise ValueError(f"Unknown sample dataset: {name}. Available: {list(self.SAMPLE_DATASETS.keys())}")
        
        text = self.SAMPLE_DATASETS[name]
        return self.load_text(text)
    
    def load_file(self, filepath: str) -> 'TextDataset':
        """Load dataset from a text file."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        return self.load_text(text)
    
    def load_text(self, text: str) -> 'TextDataset':
        """Load dataset from a string."""
        # Simple character-level tokenization
        tokens = self._tokenize(text)
        
        # Build vocabulary
        unique_chars = sorted(set(tokens))
        self.vocab = {char: i for i, char in enumerate(unique_chars)}
        self.inverse_vocab = {i: char for char, i in self.vocab.items()}
        
        # Convert to integers
        self.data = np.array([self.vocab[char] for char in tokens], dtype=np.int32)
        
        # Split into train/val
        split_idx = int(len(self.data) * self.config.train_split)
        self.train_data = self.data[:split_idx]
        self.val_data = self.data[split_idx:]
        
        return self
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into characters."""
        # Simple character-level tokenization
        # Limit to ASCII characters
        return [c for c in text if ord(c) < self.config.vocab_size]
    
    def encode(self, text: str) -> np.ndarray:
        """Encode text to token IDs."""
        tokens = self._tokenize(text)
        return np.array([self.vocab.get(char, 0) for char in tokens], dtype=np.int32)
    
    def decode(self, token_ids: np.ndarray) -> str:
        """Decode token IDs to text."""
        return ''.join([self.inverse_vocab.get(int(id), '?') for id in token_ids])
    
    def get_batch(self, split: str = 'train', batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a batch of data.
        
        Returns:
            x: Input tokens (batch_size, seq_length)
            y: Target tokens (batch_size, seq_length)
        """
        data = self.train_data if split == 'train' else self.val_data
        
        if len(data) < self.config.seq_length + 1:
            raise ValueError(f"Not enough data. Need at least {self.config.seq_length + 1} tokens.")
        
        # Random starting positions
        ix = np.random.randint(0, len(data) - self.config.seq_length, size=batch_size)
        
        x = np.stack([data[i:i+self.config.seq_length] for i in ix])
        y = np.stack([data[i+1:i+self.config.seq_length+1] for i in ix])
        
        return x, y
    
    def iter_batches(self, split: str = 'train', batch_size: int = 32) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Iterate over batches indefinitely."""
        while True:
            yield self.get_batch(split, batch_size)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        return {
            'total_tokens': len(self.data),
            'train_tokens': len(self.train_data),
            'val_tokens': len(self.val_data),
            'vocab_size': len(self.vocab),
            'seq_length': self.config.seq_length,
        }
    
    def save(self, filepath: str):
        """Save dataset to disk."""
        np.savez(
            filepath,
            data=self.data,
            train_data=self.train_data,
            val_data=self.val_data,
            vocab_keys=list(self.vocab.keys()),
            vocab_values=list(self.vocab.values()),
        )
    
    @classmethod
    def load(cls, filepath: str) -> 'TextDataset':
        """Load dataset from disk."""
        loaded = np.load(filepath, allow_pickle=True)
        
        dataset = cls()
        dataset.data = loaded['data']
        dataset.train_data = loaded['train_data']
        dataset.val_data = loaded['val_data']
        
        # Reconstruct vocab
        vocab_keys = loaded['vocab_keys']
        vocab_values = loaded['vocab_values']
        dataset.vocab = dict(zip(vocab_keys, vocab_values))
        dataset.inverse_vocab = {v: k for k, v in dataset.vocab.items()}
        
        return dataset


class Tokenizer:
    """
    Text tokenizer supporting multiple strategies.
    """
    
    def __init__(self, strategy: str = 'character', vocab_size: int = 256):
        self.strategy = strategy
        self.vocab_size = vocab_size
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        
    def train(self, texts: List[str]):
        """Train tokenizer on texts."""
        if self.strategy == 'character':
            self._train_character(texts)
        elif self.strategy == 'word':
            self._train_word(texts)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _train_character(self, texts: List[str]):
        """Train character-level tokenizer."""
        all_chars = set()
        for text in texts:
            all_chars.update(text)
        
        # Sort for deterministic vocab
        chars = sorted(list(all_chars))[:self.vocab_size]
        
        self.vocab = {char: i for i, char in enumerate(chars)}
        self.vocab['<UNK>'] = len(self.vocab)
        self.vocab['<PAD>'] = len(self.vocab)
        
        self.inverse_vocab = {i: char for char, i in self.vocab.items()}
    
    def _train_word(self, texts: List[str]):
        """Train word-level tokenizer."""
        from collections import Counter
        
        word_counts = Counter()
        for text in texts:
            words = text.split()
            word_counts.update(words)
        
        # Most common words
        most_common = word_counts.most_common(self.vocab_size - 3)
        
        self.vocab = {word: i for i, (word, _) in enumerate(most_common)}
        self.vocab['<UNK>'] = len(self.vocab)
        self.vocab['<PAD>'] = len(self.vocab)
        self.vocab['<EOS>'] = len(self.vocab)
        
        self.inverse_vocab = {i: word for word, i in self.vocab.items()}
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        if self.strategy == 'character':
            tokens = list(text)
        elif self.strategy == 'word':
            tokens = text.split()
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        return [self.vocab.get(token, self.vocab.get('<UNK>', 0)) for token in tokens]
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        tokens = [self.inverse_vocab.get(id, '<UNK>') for id in token_ids]
        
        if self.strategy == 'character':
            return ''.join(tokens)
        elif self.strategy == 'word':
            return ' '.join(tokens)
        
        return ''


# Factory function for easy dataset creation
def load_dataset(
    source: str,
    config: Optional[DatasetConfig] = None,
    **kwargs
) -> TextDataset:
    """
    Load a dataset from various sources.
    
    Args:
        source: 'shakespeare', 'alice', 'code', or path to file
        config: Dataset configuration
        **kwargs: Additional arguments for DatasetConfig
    
    Returns:
        TextDataset instance
    """
    config = config or DatasetConfig(**kwargs)
    dataset = TextDataset(config)
    
    if source in TextDataset.SAMPLE_DATASETS:
        return dataset.load_sample(source)
    elif os.path.exists(source):
        return dataset.load_file(source)
    else:
        raise ValueError(f"Unknown source: {source}")
