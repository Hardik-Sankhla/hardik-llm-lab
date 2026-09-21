# Guide: Tokenization (Chapter 2)

**Objective**: Understand how text becomes numbers. Implement BPE from scratch, compare with tiktoken, run controlled experiments on vocabulary size effects.

---

## Prerequisites

- Read: `reference/raschka/ch02/01_main-chapter-code/ch02.ipynb`
- Reference implementation: `reference/raschka/ch02/01_main-chapter-code/dataloader.ipynb`
- Key concepts: byte-pair encoding, vocabulary, compression ratio, special tokens

---

## Your Implementation Checklist

### 1. Minimal BPE Trainer (`src/hardik_llm/tokenizer/bpe.py`)

Implement these functions **from scratch** (no peeking at reference):

```python
def get_stats(ids: list[int]) -> dict[tuple[int, int], int]:
    """Count frequency of adjacent pairs."""
    pass

def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace all occurrences of pair with new_id."""
    pass

class BPETokenizer:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.merges = {}  # (int, int) -> int
        self.vocab = {}   # int -> bytes
    
    def train(self, text: str, verbose: bool = False):
        """Train BPE on text. Build merges and vocab."""
        pass
    
    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs using learned merges."""
        pass
    
    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back to text."""
        pass
    
    def save(self, path: str):
        """Save merges and vocab to JSON."""
        pass
    
    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """Load from JSON."""
        pass
```

### 2. Tokenizer Config (`src/hardik_llm/tokenizer/vocabulary.py`)

```python
@dataclass
class TokenizerConfig:
    vocab_size: int = 50257
    special_tokens: dict[str, int] = field(default_factory=lambda: {
        "<|endoftext|>": 50256,
        "<|pad|>": 50257,
    })
    pattern: str = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```

### 3. Encoding Utilities (`src/hardik_llm/tokenizer/encoding.py`)

```python
def encode_batch(tokenizer: BPETokenizer, texts: list[str], 
                 max_length: int, padding: bool = True) -> torch.Tensor:
    """Batch encode with padding/truncation."""
    pass

def decode_batch(tokenizer: BPETokenizer, ids: torch.Tensor) -> list[str]:
    """Batch decode."""
    pass
```

---

## Experiments to Run (Guide/01-tokenization/)

### EXP-001: Vocabulary Size vs Compression

**Question**: How does vocab size affect compression ratio and sequence length?

**Variables**:
- Vocab sizes: [1000, 2000, 4000, 8000, 16000, 32000, 50257]
- Dataset: 10MB sample (Wikitext-2 or your own text)

**Metrics to Measure**:
| Metric | How to Compute |
|--------|----------------|
| Compression ratio | `len(text_bytes) / len(token_ids)` |
| Avg tokens per word | `len(token_ids) / word_count` |
| Sequence length (fixed chars) | Encode 1000 chars, count tokens |
| Vocab coverage | `% of test tokens in vocab` |
| Training time | Seconds to train BPE |

**Procedure**:
1. For each vocab size, train BPE on same 10MB corpus
2. Encode fixed test set (1000 documents)
3. Record all metrics
4. Plot: vocab_size vs compression_ratio, vocab_size vs seq_length

**Expected Observation**: Larger vocab → shorter sequences but more embedding params

**Deliverable**: `experiments/results/EXP-001-tokenizer-vocab-size.json`, plots in `experiments/plots/EXP-001/`

---

### EXP-002: Your BPE vs tiktoken

**Question**: How does your implementation compare to production tiktoken?

**Variables**:
- Same vocab size (50257)
- Same training data

**Metrics**:
- Encode speed (tokens/sec)
- Decode speed
- Token ID agreement on test set
- Memory usage

**Procedure**:
1. Train your BPE to 50257 merges
2. Compare token IDs on 1000 test sentences
3. Benchmark encode/decode throughput

**Deliverable**: Comparison table, note differences

---

### EXP-003: Special Tokens Handling

**Question**: How do special tokens affect downstream training?

**Test Cases**:
1. With `<|endoftext|>` vs without
2. With `<|pad|>` for batching
3. Multiple special tokens (bos, eos, pad, unk)

**Observe**:
- Does model learn to generate `<|endoftext|>` at sequence end?
- Padding token gradient behavior
- Impact on loss at sequence boundaries

---

## What to Observe & Note

### During Implementation
- [ ] How many merges needed for 90% compression?
- [ ] What happens with rare Unicode characters?
- [ ] How does `pattern` regex affect tokenization?
- [ ] Edge cases: empty string, only special tokens, very long words

### During Experiments
- [ ] Plot vocab_size vs compression_ratio (should be diminishing returns)
- [ ] Plot vocab_size vs sequence_length (should decrease)
- [ ] Note: At what vocab size does compression plateau?
- [ ] Compare your token IDs with tiktoken on same text - where do they diverge?

### Failure Modes to Test
- [ ] Empty training text → should handle gracefully
- [ ] Vocab size < 256 → should error clearly
- [ ] Text with bytes not in training → how does encode handle?
- [ ] Very long sequence → memory behavior

---

## Validation Tests (Write in `tests/test_tokenizer.py`)

```python
def test_bpe_roundtrip():
    """encode(decode(text)) == text for various inputs."""
    pass

def test_bpe_vocab_size():
    """Trained vocab has exactly vocab_size entries."""
    pass

def test_bpe_special_tokens():
    """Special tokens encode/decode correctly."""
    pass

def test_bpe_deterministic():
    """Same training data → same merges (deterministic)."""
    pass

def test_bpe_vs_tiktoken():
    """Compare token IDs on common texts (allow small divergence)."""
    pass

def test_encode_batch_shapes():
    """Batch encode returns [B, T] tensor with correct padding."""
    pass
```

---

## Documentation to Write

1. `docs/foundations/tokenization.md` - Your explanation of BPE with diagrams
2. `guide/01-tokenization/EXPERIMENT_LOG.md` - Your running log
3. `experiments/configs/EXP-001-tokenizer-vocab-size.yaml` - Reproducible config
4. Update `docs/research/experiments.md` with EXP-001, EXP-002, EXP-003

---

## Next Step

After completing tokenizer:
1. Run all tests: `pytest tests/test_tokenizer.py -v`
2. Document findings in `docs/foundations/tokenization.md`
3. Move to `guide/02-embeddings/`