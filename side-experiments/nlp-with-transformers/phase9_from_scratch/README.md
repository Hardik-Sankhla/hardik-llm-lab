# Phase 9: Training from Scratch — Chapter 10

> **Book**: Ch 10 "Training Transformers from Scratch" (pp. 323–367)  
> **Core**: Code tokenizer (Python), custom corpus, BPE training, pretraining loop, scaling

## Book Implementation Checklist

| Section | Book Task | Key Code | Extension Ideas |
|---------|-----------|----------|-----------------|
| 10.1 | Large datasets | `load_dataset("codeparrot", streaming=True)` | The Stack, GitHub Code, RedPajama |
| 10.2 | Building custom corpus | Filter, deduplicate, shard | Quality filters, PII removal |
| 10.3 | Large dataset processing | `dataset.map(batched=True, num_proc)` | Streaming, parquet, webdataset |
| 10.4 | Train tokenizer | `tokenizers.ByteLevelBPETokenizer` | BPE vs Unigram vs WordPiece |
| 10.5 | Tokenizer evaluation | Fertility, compression, coverage | Compare vocab sizes |
| 10.6 | Pretraining objectives | MLM, CLM, PLM, span corruption | UL2, T5, mixture |
| 10.7 | Model initialization | `AutoConfig`, `AutoModelForCausalLM` | Custom config, weight init |
| 10.8 | DataLoader | `DataCollatorForLanguageModeling` | Packing, curriculum |
| 10.9 | Training loop | `Trainer` + `TrainingArguments` | `accelerate`, FSDP, DeepSpeed |
| 10.10 | Training run | Logs, checkpoints, eval | WandB, TensorBoard, scaling |
| 10.11 | Results analysis | Loss curves, perplexity, eval | Scaling laws, ablation |

## Recommended Experiments

| Exp ID | Variant | Corpus | Model | Objective | Target |
|--------|---------|--------|-------|-----------|--------|
| PH9-001 | Tiny BERT | WikiText-103 | 4L/4H/256D | MLM | Perplexity < 20 |
| PH9-002 | Tiny GPT | WikiText-103 | 4L/4H/256D | CLM | Perplexity < 25 |
| PH9-003 | Code BPE | Python (The Stack) | 50K vocab | BPE | Fertility < 1.5 |
| PH9-004 | Code GPT | Python | 6L/8H/512D | CLM | Perplexity < 10 |
| PH9-005 | Scaling sweep | 100M→1B tokens | 10M→100M params | CLM | Scaling law fit |
| PH9-006 | Objective compare | Same data | Same arch | MLM vs CLM | Downstream gap |

## Quick Start

```bash
# Book: Train BPE tokenizer (Ch 10.4)
python -m phase9_from_scratch.run \
  --config configs/phase9/tokenizer.yaml \
  --corpus python \
  --vocab-size 50000 \
  --exp-id PH9-003

# Book: Tiny BERT pretraining (Ch 10.6-10.9)
python -m phase9_from_scratch.run \
  --config configs/phase9/tiny-bert.yaml \
  --corpus wikitext-103 \
  --exp-id PH9-001

# Book: Tiny GPT pretraining (Ch 10.6-10.9)
python -m phase9_from_scratch.run \
  --config configs/phase9/tiny-gpt.yaml \
  --corpus wikitext-103 \
  --exp-id PH9-002

# Book: Scaling experiment (Ch 10.11)
python -m phase9_from_scratch.run \
  --config configs/phase9/scaling.yaml \
  --sweep tokens,params \
  --exp-id PH9-005
```

## Key Book Code Patterns to Reproduce

```python
# 1. Streaming large dataset (Ch 10.1-10.3)
from datasets import load_dataset
ds = load_dataset("codeparrot", "python", streaming=True, split="train")
# Process in streaming mode
ds = ds.shuffle(buffer_size=10000)
ds = ds.map(tokenize, batched=True, remove_columns=["code"])

# 2. Train BPE tokenizer (Ch 10.4)
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

tokenizer = ByteLevelBPETokenizer()
tokenizer.train_from_iterator(
    iterator=corpus_iterator(),
    vocab_size=50000,
    min_frequency=2,
    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
)
tokenizer.save_model("tokenizer/")

# 3. Tokenizer evaluation (Ch 10.5)
def evaluate_tokenizer(tokenizer, texts):
    total_chars = sum(len(t) for t in texts)
    total_tokens = sum(len(tokenizer.encode(t).ids) for t in texts)
    fertility = total_tokens / total_chars
    compression = total_chars / total_tokens
    return {"fertility": fertility, "compression": compression}

# 4. Model config (Ch 10.7)
from transformers import BertConfig, GPT2Config

# Tiny BERT
config = BertConfig(
    vocab_size=50000,
    hidden_size=256,
    num_hidden_layers=4,
    num_attention_heads=4,
    intermediate_size=1024,
    max_position_embeddings=512,
)
model = BertForMaskedLM(config)

# Tiny GPT
config = GPT2Config(
    vocab_size=50000,
    n_embd=256,
    n_layer=4,
    n_head=4,
    n_inner=1024,
    n_positions=512,
)
model = GPT2LMHeadModel(config)

# 5. Data collator (Ch 10.8)
from transformers import DataCollatorForLanguageModeling

# MLM (BERT)
mlm_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

# CLM (GPT)
clm_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# 6. Training arguments (Ch 10.9)
from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="./tiny-bert",
    per_device_train_batch_size=32,
    gradient_accumulation_steps=4,
    num_train_epochs=10,
    learning_rate=1e-4,
    weight_decay=0.01,
    warmup_steps=1000,
    logging_steps=100,
    save_steps=5000,
    evaluation_strategy="steps",
    eval_steps=5000,
    fp16=True,
    report_to="wandb",
    run_name="tiny-bert-wikitext",
)

# 7. Trainer (Ch 10.9)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["validation"],
    data_collator=mlm_collator,
)
trainer.train()
```

## Pretraining Objectives Comparison

| Objective | Models | Masking | Best For |
|-----------|--------|---------|----------|
| **MLM** | BERT, RoBERTa | 15% random | Encoder, understanding |
| **CLM** | GPT, Llama | Causal (left) | Decoder, generation |
| **PLM** | XLNet | Permutation | Best of both |
| **Span corruption** | T5, UL2 | Random spans | Encoder-decoder |
| **Mixture (UL2)** | UL2, Flan-UL2 | Multiple | Universal |

## Scaling Laws (Book Ch 11.1, Ref: Kaplan et al.)

```
L(N, D) = E + A/N^α + B/D^β

N = parameters, D = tokens
Optimal: N ∝ D^0.74  (Chinchilla: N ∝ D)
```

| Budget | Optimal Params | Optimal Tokens | Book Reference |
|--------|----------------|----------------|----------------|
| 10^18 FLOPs | 400M | 8B | Ch 11.1 |
| 10^21 FLOPs | 10B | 200B | Ch 11.1 |
| 10^24 FLOPs | 175B | 3.5T | Ch 11.1 |

## Evaluation Protocol

| Metric | Purpose |
|--------|---------|
| **Train loss** | Convergence, stability |
| **Eval perplexity** | Generalization |
| **Downstream FT** | Transfer quality (GLUE, etc.) |
| **Scaling law fit** | Predict larger runs |
| **Token fertility** | Tokenizer efficiency |

## Modern Extensions (Beyond Book)

| Technique | Library | Benefit |
|-----------|---------|---------|
| **FlashAttention** | `flash-attn` | 2-4x speed, less memory |
| **FSDP / DeepSpeed** | `accelerate`, `deepspeed` | Multi-node, ZeRO |
| **Curriculum learning** | Custom | Faster convergence |
| **Mixture of Experts** | `megablocks`, `fmoe` | Sparse scaling |
| **Megatron-LM** | NVIDIA | 3D parallelism |

## Next Phase Dependencies

- Phase 7 (Efficiency): Distill/quantize custom models
- Phase 10 (Frontiers): Apply to multimodal, sparse attention
- Tracks/PEFT: LoRA on custom pretrained models
- Tracks/Deployment: Export custom models to ONNX/GGUF