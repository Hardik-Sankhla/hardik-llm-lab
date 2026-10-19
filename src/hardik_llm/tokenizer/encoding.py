"""Encoding utilities for batch processing.

Implement in guide/01-tokenization/
"""

from __future__ import annotations

import torch

from hardik_llm.tokenizer.bpe import BPETokenizer


def encode_batch(
    tokenizer: BPETokenizer,
    texts: list[str],
    max_length: int,
    padding: bool = True,
    truncation: bool = True,
) -> torch.Tensor:
    """Batch encode with padding/truncation."""
    raise NotImplementedError("Implement in guide/01-tokenization/")


def decode_batch(tokenizer: BPETokenizer, ids: torch.Tensor) -> list[str]:
    """Batch decode."""
    raise NotImplementedError("Implement in guide/01-tokenization/")