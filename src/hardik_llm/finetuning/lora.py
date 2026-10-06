"""Fine-tuning utilities: classification, instruction tuning, LoRA.

Implement in guide/06-finetuning/ and guide/07-lora/
"""

import torch
import torch.nn as nn
import copy
from dataclasses import dataclass, field
from typing import List


# ============================================================
# LoRA Implementation
# ============================================================

@dataclass
class LoRAConfig:
    rank: int = 4
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: List[str] = field(default_factory=lambda: [
        "qkv_proj", "out_proj", "fc1", "fc2"
    ])


class LoRALinear(nn.Module):
    """Low-Rank Adaptation applied to a linear layer."""
    
    def __init__(self, base_layer: nn.Linear, rank: int = 4, alpha: float = 1.0, dropout: float = 0.0):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Freeze base layer
        for param in base_layer.parameters():
            param.requires_grad = False
        
        in_features = base_layer.in_features
        out_features = base_layer.out_features
        
        # LoRA adapters: A (in_features x rank), B (rank x out_features)
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        # Initialize A with normal, B with zeros (so initial output = base only)
        nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
        nn.init.zeros_(self.lora_B)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base output
        base_out = self.base_layer(x)
        
        # LoRA path: x @ A @ B * scaling
        lora_out = (self.dropout(x) @ self.lora_A @ self.lora_B) * self.scaling
        
        return base_out + lora_out
    
    def merge_weights(self):
        """Merge LoRA into base layer for inference (no runtime overhead)."""
        with torch.no_grad():
            self.base_layer.weight += (self.lora_B @ self.lora_A.T).T * self.scaling
        # Remove LoRA params
        del self.lora_A, self.lora_B


def inject_lora(model: nn.Module, config: LoRAConfig) -> nn.Module:
    """Replace target linear layers with LoRALinear."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check if this module should get LoRA
            if any(target in name for target in config.target_modules):
                parent = get_parent_module(model, name)
                child_name = name.split('.')[-1]
                setattr(parent, child_name, LoRALinear(module, config.rank, config.alpha, config.dropout))
    return model


def get_parent_module(model: nn.Module, name: str) -> nn.Module:
    parts = name.split('.')
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent


def count_lora_params(model: nn.Module) -> tuple[int, int]:
    """Returns (lora_params, total_params)"""
    lora_params = sum(p.numel() for n, p in model.named_parameters() 
                      if 'lora_' in n and p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    return lora_params, total_params


def save_lora_weights(model: nn.Module, path: str):
    """Save only LoRA parameters (tiny checkpoint)."""
    lora_state = {n: p for n, p in model.named_parameters() if 'lora_' in n}
    torch.save(lora_state, path)


def load_lora_weights(model: nn.Module, path: str):
    """Load LoRA parameters into model."""
    lora_state = torch.load(path)
    model.load_state_dict(lora_state, strict=False)


# ============================================================
# Classification Fine-tuning
# ============================================================

class ClassificationHead(nn.Module):
    """Classification head replacing LM head."""
    
    def __init__(self, d_model: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D] -> use last token (GPT-style)
        cls_token = x[:, -1, :]
        return self.classifier(self.dropout(cls_token))


def create_classification_model(base_model: nn.Module, num_classes: int, 
                                freeze_backbone: bool = True) -> nn.Module:
    """Replace LM head with classification head."""
    model = copy.deepcopy(base_model)
    model.lm_head = ClassificationHead(model.config.d_model, num_classes)
    
    if freeze_backbone:
        for param in model.embeddings.parameters():
            param.requires_grad = False
        for block in model.blocks:
            for param in block.parameters():
                param.requires_grad = False
        for param in model.ln_f.parameters():
            param.requires_grad = False
    
    return model


def freeze_strategy(model: nn.Module, strategy: str):
    """
    Strategies:
    - 'full': Train all params
    - 'head_only': Only classification head
    - 'last_n_layers': Last N transformer blocks + head
    - 'lora': Only LoRA adapters
    """
    # First freeze all
    for param in model.parameters():
        param.requires_grad = False
    
    if strategy == 'full':
        for param in model.parameters():
            param.requires_grad = True
    
    elif strategy == 'head_only':
        for param in model.lm_head.parameters():
            param.requires_grad = True
    
    elif strategy.startswith('last_'):
        n = int(strategy.split('_')[1])
        # Unfreeze last N blocks + ln_f + head
        for block in model.blocks[-n:]:
            for param in block.parameters():
                param.requires_grad = True
        for param in model.ln_f.parameters():
            param.requires_grad = True
        for param in model.lm_head.parameters():
            param.requires_grad = True
    
    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Strategy '{strategy}': {trainable:,}/{total:,} params trainable ({100*trainable/total:.1f}%)")