"""Complete GPT Model.

Implement in guide/04-transformer/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from hardik_llm.model.config import GPTConfig
from hardik_llm.model.embeddings import GPTEmbeddings
from hardik_llm.model.block import TransformerBlock


class GPTModel(nn.Module):
    """GPT-style Language Model."""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        self.embeddings = GPTEmbeddings(config)
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        self.ln_f = nn.LayerNorm(config.d_model)
        
        # LM Head (tied with token embeddings if config.tie_weights)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        if config.tie_weights:
            self.lm_head.weight = self.embeddings.token_emb.embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor = None):
        # input_ids: [B, T]
        x = self.embeddings(input_ids)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)  # [B, T, V]
        
        loss = None
        if targets is not None:
            # Shift for next-token prediction
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        return logits, loss
    
    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, 
                 temperature: float = 1.0, top_k: int = None):
        """Autoregressive generation."""
        for _ in range(max_new_tokens):
            # Crop to max_seq_len
            idx_cond = input_ids[:, -self.config.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids
    
    def get_num_params(self, non_embedding: bool = True):
        """Return number of parameters."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and self.config.tie_weights:
            # Don't count tied embeddings twice
            n_params -= self.embeddings.token_emb.embedding.weight.numel()
        return n_params