"""Encoding utilities for batch processing.

Implement in guide/01-tokenization/
"""

import torch


def encode_batch(tokenizer, texts: list[str], max_length: int, 
                 padding: bool = True, truncation: bool = True) -> torch.Tensor:
    """Batch encode with padding/truncation."""
    raise NotImplementedError("Implement in guide/01-tokenization/")


def decode_batch(tokenizer, ids: torch.Tensor) -> list[str]:
    """Batch decode."""
    raise NotImplementedError("Implement in guide/01-tokenization/")