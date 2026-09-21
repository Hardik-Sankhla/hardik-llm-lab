"""Evaluation utilities: perplexity, generation, benchmarks.

Implement in guide/08-evaluation/
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import math
from typing import list


@torch.no_grad()
def evaluate_perplexity(model: nn.Module, dataloader: DataLoader, device: str) -> float:
    """Compute perplexity on validation set."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits, loss = model(x, y)
        total_loss += loss.item() * x.numel()
        total_tokens += x.numel()
    
    model.train()
    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)


@torch.no_grad()
def evaluate_generation(
    model: nn.Module,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    device: str = "cuda"
) -> list[dict]:
    """Generate text for given prompts."""
    model.eval()
    results = []
    
    for prompt in prompts:
        input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
        generated = model.generate(input_ids, max_new_tokens, temperature, top_k)
        text = tokenizer.decode(generated[0].tolist())
        
        results.append({
            'prompt': prompt,
            'generated': text,
            'tokens_generated': max_new_tokens,
        })
    
    model.train()
    return results


@torch.no_grad()
def evaluate_next_token_accuracy(model: nn.Module, dataloader: DataLoader, device: str) -> float:
    """Next-token prediction accuracy on validation set."""
    model.eval()
    correct = 0
    total = 0
    
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        preds = logits.argmax(dim=-1)
        mask = y != -1  # Ignore padding
        correct += (preds[mask] == y[mask]).sum().item()
        total += mask.sum().item()
    
    model.train()
    return correct / total if total > 0 else 0.0