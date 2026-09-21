"""Tests for the hardik_llm package."""

import pytest
import torch


def test_import():
    """Test that package imports work."""
    import hardik_llm
    assert hardik_llm.__version__ == "0.1.0"


def test_tokenizer_import():
    """Test tokenizer imports."""
    from hardik_llm.tokenizer import BPETokenizer
    assert BPETokenizer is not None


def test_model_import():
    """Test model imports."""
    from hardik_llm.model import GPTConfig, GPTModel
    assert GPTConfig is not None
    assert GPTModel is not None


def test_attention_import():
    """Test attention imports."""
    from hardik_llm.attention import MultiHeadAttention, CausalSelfAttention
    assert MultiHeadAttention is not None
    assert CausalSelfAttention is not None


# Shape tests (to be implemented as modules are built)
class TestShapes:
    """Shape validation tests."""
    
    def test_token_embedding_shape(self):
        """TokenEmbedding: [B, T] -> [B, T, D]"""
        pytest.skip("Not implemented yet")
    
    def test_attention_shape(self):
        """MultiHeadAttention: [B, T, D] -> [B, T, D]"""
        pytest.skip("Not implemented yet")
    
    def test_gpt_forward_shape(self):
        """GPTModel: [B, T] -> logits [B, T, V]"""
        pytest.skip("Not implemented yet")
    
    def test_generation_shape(self):
        """generate: [B, T] -> [B, T + max_new_tokens]"""
        pytest.skip("Not implemented yet")


# Numerical tests (to be implemented)
class TestNumerical:
    """Numerical correctness tests."""
    
    def test_manual_vs_pytorch_sdpa(self):
        """Manual attention matches F.scaled_dot_product_attention"""
        pytest.skip("Not implemented yet")
    
    def test_weight_tying(self):
        """lm_head.weight is token_emb.embedding.weight"""
        pytest.skip("Not implemented yet")
    
    def test_lora_initial_output(self):
        """LoRA initial output equals base model"""
        pytest.skip("Not implemented yet")


# Gradient tests (to be implemented)
class TestGradients:
    """Gradient flow tests."""
    
    def test_attention_gradient_flow(self):
        """Gradients flow through Q, K, V projections"""
        pytest.skip("Not implemented yet")
    
    def test_residual_gradient_flow(self):
        """Residual connections preserve gradient flow"""
        pytest.skip("Not implemented yet")
    
    def test_lora_freeze_base(self):
        """Base layer params frozen, only LoRA trainable"""
        pytest.skip("Not implemented yet")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])