"""
hardik_llm - A from-scratch LLM research laboratory.

This package implements tokenization, attention mechanisms, transformer architectures,
training pipelines, fine-tuning, and evaluation from first principles.
"""

__version__ = "0.1.0"
__author__ = "Hardik Sankhla"
__email__ = "datascientist.hardiksankhla@gmail.com"

# Package-level exports
from hardik_llm.tokenizer import BPETokenizer
from hardik_llm.attention import MultiHeadAttention, CausalSelfAttention
from hardik_llm.model import GPTConfig, GPTModel
from hardik_llm.training import Trainer, TrainingConfig
from hardik_llm.finetuning import LoRAConfig, LoRALinear
from hardik_llm.evaluation import evaluate_perplexity, evaluate_generation

__all__ = [
    "BPETokenizer",
    "MultiHeadAttention",
    "CausalSelfAttention",
    "GPTConfig",
    "GPTModel",
    "Trainer",
    "TrainingConfig",
    "LoRAConfig",
    "LoRALinear",
    "evaluate_perplexity",
    "evaluate_generation",
]