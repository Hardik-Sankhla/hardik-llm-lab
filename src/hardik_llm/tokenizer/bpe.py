"""BPE Tokenizer implementation.

Implement following guide/01-tokenization/README.md
"""

import json
import re
from dataclasses import dataclass, field
from typing import Optional, List
import torch


@dataclass
class TokenizerConfig:
    vocab_size: int = 50257
    special_tokens: dict[str, int] = field(default_factory=lambda: {
        "<|endoftext|>": 50256,
        "<|pad|>": 50257,
    })
    # GPT-2 compatible regex pattern
    pattern: str = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class BPETokenizer:
    """Byte-Pair Encoding Tokenizer.

    Implements BPE from first principles following the guide.
    """

    def __init__(self, config: TokenizerConfig):
        self.config = config
        self.vocab_size = config.vocab_size
        self.special_tokens = config.special_tokens
        self.merges = {}  # (int, int) -> int
        self.vocab = {}   # int -> bytes
        self._trained = False
        self._pattern = None
        self._special_token_ids = set(config.special_tokens.values())
        self._id_to_special = {v: k for k, v in config.special_tokens.items()}

    def _compile_pattern(self):
        """Compile the regex pattern for tokenization."""
        # Use regex module if available for \p{L} support, else fallback
        try:
            import regex
            self._pattern = regex.compile(self.config.pattern)
        except ImportError:
            # Fallback to basic pattern without Unicode properties
            self._pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+""")

    def train(self, text: str, verbose: bool = False):
        """Train BPE on text. Build merges and vocab."""
        if verbose:
            print(f"Training BPE on {len(text):,} characters...")

        # Initialize vocab with byte tokens (0-255)
        self.vocab = {i: bytes([i]) for i in range(256)}
        next_id = 256

        # Add special tokens to vocab
        for token, token_id in self.special_tokens.items():
            self.vocab[token_id] = token.encode('utf-8')

        # Encode text to byte IDs
        ids = list(text.encode('utf-8'))

        # Train merges until vocab_size reached
        num_merges = self.vocab_size - 256 - len(self.special_tokens)

        for i in range(num_merges):
            stats = get_stats(ids)
            if not stats:
                break

            # Find most frequent pair
            pair = max(stats.items(), key=lambda x: x[1])[0]
            freq = stats[pair]

            if freq < 2:
                if verbose:
                    print(f"  Stopping: most frequent pair occurs only {freq} time(s)")
                break

            # Create new token
            new_id = next_id
            next_id += 1

            # Record merge
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]

            # Apply merge
            ids = merge(ids, pair, new_id)

            if verbose and (i + 1) % 100 == 0:
                print(f"  Merge {i + 1}/{num_merges}: {pair} -> {new_id} (freq={freq})")

        self._trained = True
        self._compile_pattern()

        if verbose:
            print(f"Training complete. Vocab size: {len(self.vocab)}, Merges: {len(self.merges)}")

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs using learned merges."""
        if not self._trained:
            raise RuntimeError("Tokenizer not trained. Call train() first.")

        if not text:
            return []

        # Handle special tokens in text
        ids = []
        for token, token_id in self.special_tokens.items():
            if token in text:
                parts = text.split(token)
                for i, part in enumerate(parts):
                    if part:
                        ids.extend(self._encode_ordinary(part))
                    if i < len(parts) - 1:
                        ids.append(token_id)
                return ids

        return self._encode_ordinary(text)

    def _encode_ordinary(self, text: str) -> List[int]:
        """Encode text without special tokens."""
        assert self._pattern is not None, "Pattern not compiled. Call train() first."

        # Split by pattern
        chunks = self._pattern.findall(text)
        ids = []

        for chunk in chunks:
            # Encode chunk to bytes
            chunk_bytes = chunk.encode('utf-8')
            chunk_ids = list(chunk_bytes)

            # Apply merges
            chunk_ids = self._apply_merges(chunk_ids)
            ids.extend(chunk_ids)

        return ids

    def _apply_merges(self, ids: List[int]) -> List[int]:
        """Apply all learned merges to a list of token IDs."""
        if not self.merges:
            return ids

        # Apply merges in order (earliest first)
        for pair, new_id in sorted(self.merges.items(), key=lambda x: x[1]):
            ids = merge(ids, pair, new_id)
        return ids

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back to text."""
        if not ids:
            return ""

        # Convert IDs to bytes
        parts = []
        for token_id in ids:
            if token_id in self._id_to_special:
                # Special token - encode its string representation
                parts.append(self._id_to_special[token_id].encode('utf-8'))
            elif token_id in self.vocab:
                parts.append(self.vocab[token_id])
            else:
                # Unknown token - use replacement character
                parts.append(b'\xef\xbf\xbd')

        text_bytes = b''.join(parts)
        return text_bytes.decode('utf-8', errors='replace')

    def save(self, path: str):
        """Save tokenizer to JSON file."""
        # Ensure special tokens are in vocab for saving
        vocab_to_save = self.vocab.copy()
        for token, token_id in self.special_tokens.items():
            vocab_to_save[token_id] = token.encode('utf-8')

        data = {
            'vocab_size': self.vocab_size,
            'special_tokens': self.special_tokens,
            'pattern': self.config.pattern,
            'merges': {f"{a},{b}": v for (a, b), v in self.merges.items()},
            'vocab': {k: list(v) for k, v in vocab_to_save.items()},
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """Load tokenizer from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        config = TokenizerConfig(
            vocab_size=data['vocab_size'],
            special_tokens=data['special_tokens'],
            pattern=data['pattern'],
        )
        tokenizer = cls(config)
        tokenizer.merges = {
            tuple(map(int, k.split(','))): v for k, v in data['merges'].items()
        }
        # Convert string keys back to int (JSON keys are always strings)
        tokenizer.vocab = {int(k): bytes(v) for k, v in data['vocab'].items()}
        tokenizer._trained = True
        tokenizer._compile_pattern()
        return tokenizer


def get_stats(ids: List[int]) -> dict[tuple[int, int], int]:
    """Count frequency of adjacent pairs."""
    stats = {}
    for i in range(len(ids) - 1):
        pair = (ids[i], ids[i + 1])
        stats[pair] = stats.get(pair, 0) + 1
    return stats


def merge(ids: List[int], pair: tuple[int, int], new_id: int) -> List[int]:
    """Replace all occurrences of pair with new_id."""
    if len(ids) < 2:
        return ids

    result = []
    i = 0
    while i < len(ids) - 1:
        if ids[i] == pair[0] and ids[i + 1] == pair[1]:
            result.append(new_id)
            i += 2
        else:
            result.append(ids[i])
            i += 1
    if i < len(ids):
        result.append(ids[i])
    return result


# Alias for backward compatibility
encode_batch = lambda tokenizer, texts, max_length, padding=True: None
decode_batch = lambda tokenizer, ids: None