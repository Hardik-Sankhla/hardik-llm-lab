"""Optimizer and scheduler creation.

Implement in guide/05-pretraining/
"""

import torch
import torch.nn as nn
import math


def create_optimizer(model: nn.Module, config: "TrainingConfig"):
    """Create AdamW optimizer with proper weight decay grouping."""
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # No weight decay on bias, LayerNorm, embeddings
        if ('bias' in name or 
            'ln' in name.lower() or 
            'norm' in name.lower() or 
            'embedding' in name.lower() or
            'pos_emb' in name):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    optimizer = torch.optim.AdamW(
        [
            {'params': decay_params, 'weight_decay': config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ],
        lr=config.learning_rate, 
        betas=(config.beta1, config.beta2), 
        eps=config.eps
    )
    return optimizer


def create_scheduler(optimizer, config: "TrainingConfig"):
    """Create cosine decay with warmup scheduler."""
    def lr_lambda(step):
        if step < config.warmup_steps:
            return step / config.warmup_steps
        # Cosine decay
        progress = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
        progress = min(progress, 1.0)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return config.min_lr / config.learning_rate + (1 - config.min_lr / config.learning_rate) * cosine
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_grad_norm(model: nn.Module) -> float:
    """Compute total gradient norm."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5