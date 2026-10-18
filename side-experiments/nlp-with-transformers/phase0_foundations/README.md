# Phase 0: Foundations — Chapter 1

> **Book**: Ch 1 "Hello Transformers" (pp. 25–44)  
> **Core**: Transformer mechanics, encoder-decoder, attention, transfer learning, HF ecosystem tour

## Book Implementation Checklist

| Section | Book Task | Key Code | Extension Ideas |
|---------|-----------|----------|-----------------|
| 1.1 | Encoder-decoder framework | `AutoModelForSeq2SeqLM` | Compare encoder-only vs decoder-only |
| 1.2 | Attention mechanisms | `model.config.num_attention_heads` | Visualize attention weights |
| 1.3 | Transfer learning in NLP | Pretrain → Fine-tune paradigm | Domain-adaptive pretraining |
| 1.4 | HF Transformers library | `pipeline()`, `AutoModel`, `Trainer` | Custom models, custom heads |
| 1.5 | Application tour | Classification, NER, QA, Summ, Trans, Gen | Quick-start each task |
| 1.6 | HF Ecosystem | Hub, Tokenizers, Datasets, Accelerate | Hub upload, private repos |
| 1.7 | Challenges | Hallucination, bias, efficiency | Evaluation, mitigation |

## Recommended Experiments

| Exp ID | Variant | Focus | Key Insight |
|--------|---------|-------|-------------|
| PH0-001 | Pipeline tour | All `pipeline()` tasks | Zero-shot capabilities |
| PH0-002 | Attention viz | Encoder/decoder attention | Head specialization |
| PH0-003 | Transfer learning | Pretrain vs fine-tune | Layer freezing strategies |
| PH0-004 | Hub interaction | Push/pull models/datasets | Versioning, spaces |
| PH0-005 | Tokenizer comparison | BERT vs RoBERTa vs T5 | Subword vocab impact |

## Quick Start

```bash
# Book: Pipeline tour (Ch 1.5)
python -m phase0_foundations.run --demo pipelines --exp-id PH0-001

# Book: Attention visualization (Ch 1.2)
python -m phase0_foundations.run --demo attention --model bert-base-uncased --exp-id PH0-002

# Book: Transfer learning demo (Ch 1.3)
python -m phase0_foundations.run --demo transfer --exp-id PH0-003

# HF Hub: Push model (Ch 1.6)
python -m phase0_foundations.run --demo hub --push my-model --exp-id PH0-004
```

## Key Book Code Patterns to Reproduce

```python
# 1. Pipelines (Ch 1.5)
from transformers import pipeline

# Classification
classifier = pipeline("sentiment-analysis")
classifier("I love transformers!")

# NER
ner = pipeline("ner", aggregation_strategy="simple")
ner("Hugging Face is in Paris.")

# QA
qa = pipeline("question-answering")
qa(question="Where is Hugging Face?", context="Hugging Face is in Paris.")

# Summarization
summarizer = pipeline("summarization")
summarizer(long_text)

# Translation
translator = pipeline("translation_en_to_fr")
translator("Hello world")

# Generation
generator = pipeline("text-generation", model="gpt2")
generator("Transformers are")

# 2. AutoClasses (Ch 1.4)
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")
classifier = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

# 3. Attention visualization (Ch 1.2)
import torch
inputs = tokenizer("The cat sat on the mat", return_tensors="pt")
outputs = model(**inputs, output_attentions=True)
attentions = outputs.attentions  # Tuple of (batch, heads, seq, seq) per layer

# 4. Transfer learning pattern (Ch 1.3)
# Freeze backbone, train head
for param in model.base_model.parameters():
    param.requires_grad = False
# Only classifier head trains

# 5. HF Hub (Ch 1.6)
from huggingface_hub import HfApi, create_repo
api = HfApi()
create_repo("my-username/my-model", private=False)
model.push_to_hub("my-username/my-model")
tokenizer.push_to_hub("my-username/my-model")
```

## HF Ecosystem Components (Ch 1.6)

| Component | Purpose | Key Class/CLI |
|-----------|---------|---------------|
| **Transformers** | Models, tokenizers, pipelines | `AutoModel`, `pipeline()` |
| **Datasets** | Load, process, share datasets | `load_dataset()`, `Dataset` |
| **Tokenizers** | Fast Rust tokenizers | `AutoTokenizer`, `train_new_from_iterator()` |
| **Accelerate** | Multi-GPU/TPU, mixed precision | `Accelerator`, `launch` |
| **Hub** | Model/dataset hosting, Spaces | `huggingface_hub`, `hf` CLI |
| **Evaluate** | Metrics, benchmarks | `load("accuracy")`, `evaluate()` |
| **PEFT** | LoRA, adapters, prompt tuning | `get_peft_model()` |
| **TRL** | RLHF, DPO, PPO | `DPOTrainer`, `PPOTrainer` |

## Environment Setup (Book Ch 1.2)

```bash
# Core
pip install transformers datasets tokenizers accelerate evaluate

# Hub
pip install huggingface_hub

# PEFT/RLHF (modern)
pip install peft trl bitsandbytes

# Serving (modern)
pip install vllm optimum[onnxruntime]

# Optional: WandB for tracking
pip install wandb
```

## Next Phase Dependencies

- Phase 1 (Classification): Uses pipelines, AutoClasses, Hub
- Phase 2 (Architecture): Deep dive into attention, encoder/decoder
- All phases: Use Datasets, Tokenizers, Accelerate, Hub
- Tracks/PEFT: Builds on transfer learning paradigm
- Tracks/Deployment: Builds on efficiency challenges