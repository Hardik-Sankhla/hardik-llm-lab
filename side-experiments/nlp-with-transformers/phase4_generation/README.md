# Phase 4: Text Generation — Chapter 5

> **Book**: Ch 5 "Text Generation" (pp. 147–164)  
> **Core**: Decoding strategies — greedy, beam search, sampling, top-k, nucleus (GPT-2)

## Book Implementation Checklist

| Section | Book Task | Key Code | Extension Ideas |
|---------|-----------|----------|-----------------|
| 5.1 | Coherent text challenge | `model.generate()` defaults | Compare all decoding methods |
| 5.2 | Greedy search | `generate(do_sample=False, num_beams=1)` | Repetition analysis |
| 5.3 | Beam search | `generate(num_beams=5, early_stopping=True)` | Beam width vs quality/speed |
| 5.4 | Sampling (temp, top-k) | `generate(do_sample=True, temperature=0.7)` | Temperature sweep |
| 5.5 | Top-k & nucleus (top-p) | `generate(top_k=50, top_p=0.9)` | Top-k vs top-p tradeoffs |
| 5.6 | Best decoding method | Human eval + perplexity | Task-dependent: creative vs factual |

## Recommended Experiments

| Exp ID | Variant | Model | Decoding | Params | Target |
|--------|---------|-------|----------|--------|--------|
| PH4-001 | Greedy baseline | `gpt2` | Greedy | — | Perplexity |
| PH4-002 | Beam search | `gpt2` | Beam | beams=5 | Perplexity, diversity |
| PH4-003 | Top-k sampling | `gpt2` | Top-k | k=50 | Diversity (distinct-n) |
| PH4-004 | Nucleus sampling | `gpt2` | Top-p | p=0.9 | Quality (human eval) |
| PH4-005 | Temperature sweep | `gpt2` | Sampling | T=0.1..1.5 | Perplexity vs diversity |
| PH4-006 | Larger model | `gpt2-medium` | Nucleus | p=0.9 | Quality improvement |
| PH4-007 | Instruction-tuned | `gpt2` + LoRA on Alpaca | Nucleus | p=0.9 | Instruction following |

## Models (Book + Modern)

| Model | Type | Size | Book Ch | Modern Equivalent |
|-------|------|------|---------|-------------------|
| `gpt2` | Decoder | 124M | Ch 5 | `gpt2` (baseline) |
| `gpt2-medium` | Decoder | 355M | Ch 5 | `gpt2-medium` |
| `gpt2-large` | Decoder | 774M | — | `gpt2-large` |
| — | — | — | — | `mistral-7b` (instruction-tuned) |
| — | — | — | — | `llama-2-7b-chat` |

## Quick Start

```bash
# Book: Greedy vs Beam vs Sampling (GPT-2)
python -m phase4_generation.run \
  --config configs/phase4/gpt2-decoding.yaml \
  --method greedy \
  --exp-id PH4-001

python -m phase4_generation.run \
  --config configs/phase4/gpt2-decoding.yaml \
  --method beam --num-beams 5 \
  --exp-id PH4-002

python -m phase4_generation.run \
  --config configs/phase4/gpt2-decoding.yaml \
  --method nucleus --top-p 0.9 --temperature 0.7 \
  --exp-id PH4-004

# Temperature sweep
python -m phase4_generation.run \
  --config configs/phase4/gpt2-decoding.yaml \
  --method sweep --temperatures 0.1,0.3,0.5,0.7,1.0,1.5 \
  --exp-id PH4-005
```

## Key Book Code Patterns to Reproduce

```python
# 1. Basic generation (Ch 5.1)
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
prompt = "Transformers are"
inputs = tokenizer(prompt, return_tensors="pt")

# 2. Greedy (Ch 5.2)
outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# 3. Beam search (Ch 5.3)
outputs = model.generate(
    **inputs, max_new_tokens=50,
    num_beams=5, early_stopping=True, no_repeat_ngram_size=2
)

# 4. Sampling with temperature (Ch 5.4)
outputs = model.generate(
    **inputs, max_new_tokens=50,
    do_sample=True, temperature=0.7, top_k=0
)

# 5. Top-k sampling (Ch 5.5)
outputs = model.generate(
    **inputs, max_new_tokens=50,
    do_sample=True, top_k=50
)

# 6. Nucleus (top-p) sampling (Ch 5.5)
outputs = model.generate(
    **inputs, max_new_tokens=50,
    do_sample=True, top_p=0.9, temperature=0.7
)

# 7. Evaluate: Perplexity + Diversity (distinct-n)
def distinct_n(texts, n):
    ngrams = set()
    total = 0
    for text in texts:
        tokens = text.split()
        for i in range(len(tokens) - n + 1):
            ngrams.add(tuple(tokens[i:i+n]))
            total += 1
    return len(ngrams) / total if total > 0 else 0
```

## Decoding Strategy Comparison (Book Table 5.1)

| Method | Coherence | Diversity | Speed | Best For |
|--------|-----------|-----------|-------|----------|
| Greedy | High | Low | Fastest | Deterministic, factual |
| Beam Search | High | Medium | Slow | Translation, summarization |
| Top-k | Medium | High | Fast | Creative writing |
| Nucleus (Top-p) | High | High | Fast | General-purpose, chat |
| Temperature | Variable | Variable | Fast | Control creativity |

## Evaluation Metrics

| Metric | What It Measures | Implementation |
|--------|------------------|----------------|
| **Perplexity** | Fluency, model confidence | `torch.exp(loss)` |
| **Distinct-n** | Lexical diversity | n-gram uniqueness |
| **Repetition Rate** | Degeneration | % repeated n-grams |
| **Human Eval** | Quality, coherence | Pairwise comparison |
| **MAUVE** | Distribution similarity | `mauve` library |

## Modern Extensions (Beyond Book)

| Technique | Library | Use Case |
|-----------|---------|----------|
| **LoRA/QLoRA** | `peft` | Adapt GPT-2 to instruction following |
| **DPO/PPO** | `trl` | Align with human preferences |
| **Speculative Decoding** | `vLLM`, `transformers` | Faster generation |
| **Guidance/Outlines** | `guidance`, `outlines` | Constrained generation |
| **vLLM/TGI** | Serving | Production throughput |

## Next Phase Dependencies

- Phase 5 (Summarization): Encoder-decoder generation (BART, T5, PEGASUS)
- Phase 6 (QA/RAG): Retrieval-augmented generation
- Phase 7 (Efficiency): Quantize GPT-2 for deployment
- Tracks/PEFT: Apply LoRA to instruction-tune GPT-2