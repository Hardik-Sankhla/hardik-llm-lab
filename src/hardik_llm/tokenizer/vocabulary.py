"""Vocabulary utilities for tokenizer.

Implement in guide/01-tokenization/
"""

from dataclasses import dataclass, field


@dataclass
class Vocabulary:
    """Vocabulary mapping."""
    token_to_id: dict[str, int] = field(default_factory=dict)
    id_to_token: dict[int, str] = field(default_factory=dict)
    
    def __len__(self):
        return len(self.token_to_id)
    
    def add_token(self, token: str) -> int:
        """Add token, return its ID."""
        if token not in self.token_to_id:
            idx = len(self.token_to_id)
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
        return self.token_to_id[token]
    
    def __getitem__(self, key):
        if isinstance(key, str):
            return self.token_to_id[key]
        return self.id_to_token[key]
    
    def __contains__(self, key):
        if isinstance(key, str):
            return key in self.token_to_id
        return key in self.id_to_token