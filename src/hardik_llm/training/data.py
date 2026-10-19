"""Data loading utilities.

Implement in guide/05-pretraining/
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader, Dataset

if TYPE_CHECKING:
    from hardik_llm.training.config import TrainingConfig


class TextDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Dataset for next-token prediction from token sequences."""

    def __init__(self, tokens: list[int], max_seq_len: int) -> None:
        self.tokens = tokens
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return max(0, len(self.tokens) - self.max_seq_len)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.tokens[idx : idx + self.max_seq_len], dtype=torch.long)
        y = torch.tensor(
            self.tokens[idx + 1 : idx + 1 + self.max_seq_len], dtype=torch.long
        )
        return x, y


def create_dataloaders(
    train_tokens: list[int], val_tokens: list[int], config: TrainingConfig
) -> tuple[DataLoader[tuple[torch.Tensor, torch.Tensor]], DataLoader[tuple[torch.Tensor, torch.Tensor]]]:
    """Create train and validation dataloaders."""
    train_ds = TextDataset(train_tokens, config.max_seq_len)
    val_ds = TextDataset(val_tokens, config.max_seq_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader