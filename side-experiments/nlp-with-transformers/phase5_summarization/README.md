# Phase 5: Summarization — Chapter 6

> **Book**: Ch 6 "Summarization" (pp. 165–187)  
> **Core**: CNN/DailyMail with PEGASUS/BART/T5 — ROUGE, BLEU, fine-tuning

## Book Implementation Checklist

| Section | Book Task | Key Code | Extension Ideas |
|---------|-----------|----------|-----------------|
| 6.1 | CNN/DailyMail dataset | `load_dataset("cnn_dailymail", "3.0.0")` | Try XSum, SAMSum, arXiv |
| 6.2 | Summarization pipelines | `pipeline("summarization", model=...)` | Compare BART, T5, PEGASUS, GPT-2 |
| 6.3 | Baseline evaluation | `generate()` + ROUGE | Zero-shot vs fine-tuned |
| 6.4 | Quality metrics | ROUGE-1/2/L, BLEU | BERTScore, MoverScore |
| 6.5 | PEGASUS on CNN/DM | Fine-tune PEGASUS | LoRA for efficient FT |
| 6.6 | SAMSum dialogue | Fine-tune on dialogue | Speaker-aware summarization |
| 6.7 | Generate dialogue summaries | `model.generate()` with prompts | Controllable length/style |

## Recommended Experiments

| Exp ID | Variant | Model | Dataset | PEFT | Target ROUGE-L |
|--------|---------|-------|---------|------|----------------|
| PH5-001 | BART zero-shot | `facebook/bart-large-cnn` | CNN/DM | — | ≥ 0.35 |
| PH5-002 | PEGASUS zero-shot | `google/pegasus-cnn_dailymail` | CNN/DM | — | ≥ 0.38 |
| PH5-003 | T5 zero-shot | `t5-base` | CNN/DM | — | ≥ 0.30 |
| PH5-004 | PEGASUS fine-tuned | `google/pegasus-cnn_dailymail` | CNN/DM | Full FT | ≥ 0.43 |
| PH5-005 | PEGASUS LoRA | `google/pegasus-cnn_dailymail` | CNN/DM | LoRA r=32 | ≥ 0.41 |
| PH5-006 | BART fine-tuned | `facebook/bart-large-cnn` | CNN/DM | Full FT | ≥ 0.42 |
| PH5-007 | SAMSum dialogue | `facebook/bart-large` | SAMSum | LoRA r=16 | ≥ 0.45 |
| PH5-008 | DistilBART | `sshleifer/distilbart-cnn-12-6` | CNN/DM | Full FT | ≥ 0.39 |

## Datasets (Book + Extensions)

| Dataset | Domain | Style | Size | Book Ch |
|---------|--------|-------|------|---------|
| `cnn_dailymail` | News | Article → Highlights | 287K | Ch 6 |
| `xsum` | News | Article → 1 sentence | 204K | Ch 6 |
| `samsum` | Dialogue | Conversation → Summary | 14K | Ch 6 |
| `arxiv` / `pubmed` | Scientific | Paper → Abstract | 200K+ | — |
| `multi_news` | News | Multi-doc → Summary | 44K | — |
| `wikihow` | How-to | Article → Steps | 168K | — |

## Quick Start

```bash
# Book: Zero-shot PEGASUS on CNN/DM (Ch 6.2-6.3)
python -m phase5_summarization.run \
  --config configs/phase5/pegasus-cnn_dailymail.yaml \
  --mode eval \
  --exp-id PH5-002

# Book: Fine-tune PEGASUS (Ch 6.5)
python -m phase5_summarization.run \
  --config configs/phase5/pegasus-cnn_dailymail.yaml \
  --mode train \
  --exp-id PH5-004

# LoRA efficient fine-tuning (tracks/peft)
python -m phase5_summarization.run \
  --config configs/phase5/pegasus-cnn_dailymail.yaml \
  --peft lora --peft-r 32 \
  --exp-id PH5-005

# SAMSum dialogue (Ch 6.6-6.7)
python -m phase5_summarization.run \
  --config configs/phase5/bart-samsum.yaml \
  --exp-id PH5-007
```

## Key Book Code Patterns to Reproduce

```python
# 1. CNN/DailyMail loading (Ch 6.1)
from datasets import load_dataset
ds = load_dataset("cnn_dailymail", "3.0.0")
# Features: article, highlights, id

# 2. Preprocessing for encoder-decoder (Ch 6.2)
def preprocess(examples):
    inputs = tokenizer(examples["article"], max_length=1024, truncation=True)
    targets = tokenizer(examples["highlights"], max_length=128, truncation=True)
    inputs["labels"] = targets["input_ids"]
    return inputs

# 3. ROUGE evaluation (Ch 6.4)
from evaluate import load
rouge = load("rouge")

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # ROUGE expects newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(p)) for p in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(l)) for l in decoded_labels]
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    return {k: round(v * 100, 4) for k, v in result.items()}

# 4. PEGASUS fine-tuning (Ch 6.5)
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
model = AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-cnn_dailymail")
training_args = Seq2SeqTrainingArguments(
    output_dir="./pegasus-cnn-dm",
    evaluation_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=True,
)
trainer = Seq2SeqTrainer(
    model=model, args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
trainer.train()
```

## Model Comparison (Book Table 6.1)

| Model | Architecture | Pretraining | Best For |
|-------|--------------|-------------|----------|
| **BART** | Encoder-decoder | Denoising autoencoder | General, CNN/DM |
| **PEGASUS** | Encoder-decoder | Gap-sentence generation | News summarization |
| **T5** | Encoder-decoder | Span corruption | Multi-task, flexible |
| **GPT-2** | Decoder-only | Causal LM | Prompt-based (Ch 6.2) |

## Evaluation Metrics

| Metric | Book Coverage | Implementation |
|--------|---------------|----------------|
| **ROUGE-1** | Ch 6.4 | Unigram overlap |
| **ROUGE-2** | Ch 6.4 | Bigram overlap |
| **ROUGE-L** | Ch 6.4 | Longest common subsequence |
| **BLEU** | Ch 6.4 | Precision n-grams (MT) |
| **BERTScore** | — | Semantic similarity |
| **MoverScore** | — | Word mover distance |

## Modern Extensions (Beyond Book)

| Technique | Library | Use Case |
|-----------|---------|----------|
| **LoRA/QLoRA** | `peft` | Efficient FT of BART/PEGASUS/T5 |
| **Controlled generation** | `transformers` | Length, style, entity control |
| **Long-form** | `longformer`, `LED` | >1024 tokens (LED, BigBird) |
| **Factuality** | `SummaC`, `QuestEval` | Hallucination detection |
| **vLLM/TGI** | Serving | High-throughput summarization |

## Next Phase Dependencies

- Phase 4 (Generation): Decoder-only models for prompt-based summarization
- Phase 6 (QA/RAG): Retrieval-augmented summarization
- Phase 7 (Efficiency): Distill PEGASUS/BART, quantize for deployment
- Tracks/PEFT: LoRA critical for large encoder-decoder models