"""Tests for the hardik_llm package."""

import pytest
import torch
from hardik_llm.tokenizer import BPETokenizer, TokenizerConfig, get_stats, merge


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


# ============================================================
# BPE Tokenizer Tests (Chapter 1 - Implemented)
# ============================================================

class TestBPETokenizer:
    """Tests for BPE Tokenizer implementation."""

    def setup_method(self):
        """Create a fresh tokenizer for each test."""
        self.config = TokenizerConfig(vocab_size=1000)
        self.tokenizer = BPETokenizer(self.config)
        self.text = "Hello world! This is a test. Hello again world!"

    def test_get_stats(self):
        """Test pair frequency counting."""
        ids = [1, 2, 3, 1, 2, 4]
        stats = get_stats(ids)
        assert stats[(1, 2)] == 2
        assert stats[(2, 3)] == 1
        assert stats[(3, 1)] == 1
        assert stats[(2, 4)] == 1

    def test_merge(self):
        """Test pair merging."""
        ids = [1, 2, 3, 1, 2, 4]
        merged = merge(ids, (1, 2), 5)
        assert merged == [5, 3, 5, 4]

    def test_merge_edge_cases(self):
        """Test merge with edge cases."""
        # Empty list
        assert merge([], (1, 2), 5) == []
        # Single element
        assert merge([1], (1, 2), 5) == [1]
        # No match
        assert merge([1, 3, 4], (1, 2), 5) == [1, 3, 4]
        # Overlapping matches shouldn't happen in BPE but test anyway
        assert merge([1, 2, 2, 3], (1, 2), 5) == [5, 2, 3]

    def test_train_basic(self):
        """Test basic BPE training."""
        self.tokenizer.train(self.text, verbose=False)
        assert self.tokenizer._trained
        assert len(self.tokenizer.merges) > 0
        assert len(self.tokenizer.vocab) > 256  # At least byte vocab + merges

    def test_encode_decode_roundtrip(self):
        """Test encode -> decode roundtrip."""
        self.tokenizer.train(self.text, verbose=False)
        ids = self.tokenizer.encode(self.text)
        decoded = self.tokenizer.decode(ids)
        assert decoded == self.text

    def test_encode_empty(self):
        """Test encoding empty string."""
        self.tokenizer.train(self.text, verbose=False)
        ids = self.tokenizer.encode("")
        assert ids == []

    def test_special_tokens(self):
        """Test special token handling."""
        text_with_special = "Hello <|endoftext|> world"
        self.tokenizer.train(text_with_special, verbose=False)
        
        ids = self.tokenizer.encode(text_with_special)
        assert 50256 in ids  # <|endoftext|>
        
        decoded = self.tokenizer.decode(ids)
        assert "<|endoftext|>" in decoded

    def test_save_load(self):
        """Test save/load persistence."""
        import tempfile
        import os
        
        self.tokenizer.train(self.text, verbose=False)
        original_ids = self.tokenizer.encode(self.text)
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        
        try:
            self.tokenizer.save(path)
            loaded = BPETokenizer.load(path)
            
            # Check vocab and merges preserved
            assert len(loaded.vocab) == len(self.tokenizer.vocab)
            assert len(loaded.merges) == len(self.tokenizer.merges)
            assert loaded.config.vocab_size == self.tokenizer.config.vocab_size
            
            # Check encode/decode works after load
            loaded_ids = loaded.encode(self.text)
            assert loaded_ids == original_ids
            
            loaded_decoded = loaded.decode(loaded_ids)
            assert loaded_decoded == self.text
        finally:
            os.unlink(path)

    def test_vocab_size_limit(self):
        """Test vocab size is respected."""
        # Train on larger text
        large_text = self.text * 100
        config = TokenizerConfig(vocab_size=500)
        tokenizer = BPETokenizer(config)
        tokenizer.train(large_text, verbose=False)
        
        # Vocab should not exceed configured size (plus special tokens)
        assert len(tokenizer.vocab) <= config.vocab_size + len(config.special_tokens)

    def test_deterministic_training(self):
        """Test that same text produces same merges."""
        config = TokenizerConfig(vocab_size=500)
        tokenizer1 = BPETokenizer(config)
        tokenizer2 = BPETokenizer(config)
        
        tokenizer1.train(self.text, verbose=False)
        tokenizer2.train(self.text, verbose=False)
        
        assert tokenizer1.merges == tokenizer2.merges
        assert tokenizer1.vocab == tokenizer2.vocab


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