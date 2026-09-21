"""BPE Tokenizer implementation.

Implement in guide/01-tokenization/
"""

import json
from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class TokenizerConfig:
    vocab_size: int = 50257
    special_tokens: dict[str, int] = field(default_factory=lambda: {
        "<|endoftext|>": 50256,
        "<|pad|>": 50257,
    })
    pattern: str = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class BPETokenizer:
    """Byte-Pair Encoding Tokenizer.
    
    Implement following guide/01-tokenization/README.md
    """
    
    def __init__(self, config: TokenizerConfig):
        self.config = config
        self.vocab_size = config.vocab_size
        self.special_tokens = config.special_tokens
        self.merges = {}  # (int, int) -> int
        self.vocab = {}   # int -> bytes
        self._trained = False
    
    def train(self, text: str, verbose: bool = False):
        """Train BPE on text."""
        raise NotImplementedError("Implement in guide/01-tokenization/")
    
    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        raise NotImplementedError("Implement in guide/01-tokenization/")
    
    def decode(self, ids: list[int]) -> str:
        """Decode token IDs to text."""
        raise NotImplementedError("Implement in guide/01-tokenization/")
    
    def save(self, path: str):
        """Save to JSON."""
        raise NotImplementedError("Implement in guide/01-tokenization/")
    
    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """Load from JSON."""
        raise NotImplementedError("Implement in guide/01-tokenization/")


def get_stats(ids: list[int]) -> dict[tuple[int, int], int]:
    """Count frequency of adjacent pairs."""
    raise NotImplementedError("Implement in guide/01-tokenization/")


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace all occurrences of pair with new_id."""
    raise NotImplementedError("Implement in guide/01-tokenization/")