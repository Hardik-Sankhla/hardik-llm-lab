# Guide: Evaluation Framework (Chapter 5, 6, 7 + beyond)

**Objective**: Build comprehensive evaluation: perplexity, generation quality, benchmarks, human eval. Measure what matters.

---

## Prerequisites

- Read: `reference/raschka/ch05/01_main-chapter-code/gpt_generate.py`
- Read: `reference/raschka/ch07/01_main-chapter-code/ollama_evaluate.py`
- Key concepts: Perplexity, BLEU/ROUGE, few-shot eval, generation metrics

---

## Your Implementation Checklist

### 1. Perplexity Evaluation (`src/hardik_llm/evaluation/perplexity.py`)

```python
@torch.no_grad()
def evaluate_perplexity(model: GPTModel, dataloader: DataLoader, device: str) -> float:
    """Compute perplexity on validation set."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits, loss = model(x, y)
        # loss is already averaged over batch and seq_len
        total_loss += loss.item() * x.numel()
        total_tokens += x.numel()
    
    model.train()
    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)
```

### 2. Generation Evaluation (`src/hardik_llm/evaluation/generation.py`)

```python
@torch.no_grad()
def evaluate_generation(
    model: GPTModel,
    tokenizer: BPETokenizer,
    prompts: list[str],
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    device: str = "cuda"
) -> list[dict]:
    """Generate and return structured results."""
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
```

### 3. Benchmark Evaluation (`src/hardik_llm/evaluation/benchmarks.py`)

```python
def evaluate_benchmarks(model: GPTModel, tokenizer: BPETokenizer, device: str) -> dict:
    """Run standard benchmarks."""
    results = {}
    
    # Lambada (last word prediction)
    results['lambada'] = evaluate_lambada(model, tokenizer, device)
    
    # HellaSwag (commonsense reasoning)
    results['hellaswag'] = evaluate_hellaswag(model, tokenizer, device)
    
    # PIQA (physical reasoning)
    results['piqa'] = evaluate_piqa(model, tokenizer, device)
    
    # Custom: Next-token accuracy on validation set
    results['next_token_acc'] = evaluate_next_token_accuracy(model, tokenizer, device)
    
    return results

def evaluate_next_token_accuracy(model, tokenizer, device):
    """Simple next-token accuracy on held-out data."""
    pass
```

### 4. Human Evaluation Framework (`src/hardik_llm/evaluation/human.py`)

```python
def create_human_eval_prompts() -> list[dict]:
    """Prompts for human evaluation."""
    return [
        {"category": "factual", "prompt": "The capital of France is"},
        {"category": "reasoning", "prompt": "If all roses are flowers and some flowers are red, then"},
        {"category": "creative", "prompt": "Write a haiku about transformers:"},
        {"category": "code", "prompt": "def fibonacci(n):"},
        {"category": "instruction", "prompt": "### Instruction:\nExplain attention simply\n\n### Response:"},
    ]

def run_human_eval(model, tokenizer, device) -> pd.DataFrame:
    """Generate responses for human rating."""
    prompts = create_human_eval_prompts()
    results = evaluate_generation(model, tokenizer, [p['prompt'] for p in prompts], device=device)
    
    df = pd.DataFrame(results)
    df['category'] = [p['category'] for p in prompts]
    return df
```

---

## Experiments to Run (Guide/08-evaluation/)

### EXP-026: Perplexity vs Generation Quality Correlation

**Question**: Does lower perplexity always mean better generations?

**Variables**: Models at different training checkpoints

**Metrics**:
- Perplexity (val)
- Human eval scores (1-5)
- Distinct-n (diversity)
- Repetition rate

**Procedure**: Evaluate 10 checkpoints, plot perplexity vs human score

---

### EXP-027: Temperature & Top-k Effects

**Question**: How do sampling parameters affect quality/diversity tradeoff?

**Variables**:
- Temperature: [0.1, 0.3, 0.5, 0.7, 1.0, 1.2]
- Top-k: [1, 10, 40, 100, None]

**Metrics**:
- Perplexity of generated text
- Distinct-1, Distinct-2
- Repetition rate
- Human quality rating

---

### EXP-028: Few-shot Evaluation

**Question**: Does the model learn in-context?

**Variables**: 0-shot, 1-shot, 3-shot, 5-shot prompts

**Tasks**: Simple classification, translation, pattern completion

---

## What to Observe & Note

### During Implementation
- [ ] Perplexity: use same tokenization as training
- [ ] Generation: batch generation for efficiency
- [ ] Benchmarks: need standardized datasets (download once)

### During Experiments
- [ ] Plot perplexity vs checkpoint step
- [ ] Temperature sweep: find sweet spot
- [ ] Human eval: use consistent rubric (coherence, relevance, correctness)
- [ ] Few-shot: does performance scale with shots?

---

## Validation Tests (`tests/test_evaluation.py`)

```python
def test_perplexity_calculation():
    """Perplexity = exp(avg_cross_entropy)"""
    pass

def test_generation_deterministic():
    """temp=0 gives deterministic output."""
    pass

def test_generation_shape():
    """generate increases sequence length correctly."""
    pass
```

---

## Documentation to Write

1. `docs/research/benchmarks.md` - Benchmark details, results
2. `docs/research/evaluation.md` - Evaluation methodology
3. `guide/08-evaluation/EXPERIMENT_LOG.md`
4. Update `docs/research/experiments.md`

---

## Next Step

After evaluation: `guide/09-experiment-infra/`