# Phase 1: Text Classification — Chapter 2

> **Book**: Ch 2 "Text Classification" (pp. 45–78)  
> **Core**: Tweets sentiment classification with BERT — tokenization, feature extraction vs fine-tuning

## Book Implementation Checklist

| Section | Book Task | Key Code | Extension Ideas |
|---------|-----------|----------|-----------------|
| 2.1 | Load & explore dataset (TweetEval) | `datasets.load_dataset("tweet_eval", "sentiment")` | Try other TweetEval tasks (emotion, hate, irony) |
| 2.2 | Tokenization (BERT tokenizer) | `AutoTokenizer.from_pretrained("bert-base-uncased")` | Compare char/word/subword (Ch 2.2) |
| 2.3 | Feature extraction (frozen BERT) | `model(**inputs)[0][:, 0]` → LogisticRegression | Try different pooling (CLS, mean, max) |
| 2.4 | Fine-tuning BERT | `Trainer` with `TrainingArguments` | Add LoRA (tracks/peft), compare full vs PEFT |
| 2.5 | Error analysis | Confusion matrix, misclassified examples | Per-class metrics, data slices |

## Recommended Experiments

| Exp ID | Variant | Model | PEFT | Dataset | Target Metric |
|--------|---------|-------|------|---------|---------------|
| PH1-001 | Book baseline | `bert-base-uncased` | Full FT | TweetEval sentiment | F1 ≥ 0.70 |
| PH1-002 | Feature extractor | `bert-base-uncased` | Frozen + LR | TweetEval sentiment | F1 ≥ 0.65 |
| PH1-003 | LoRA efficient | `bert-base-uncased` | LoRA r=16 | TweetEval sentiment | F1 ≥ 0.68 |
| PH1-004 | DistilBERT | `distilbert-base-uncased` | Full FT | TweetEval sentiment | F1 ≥ 0.67 |
| PH1-005 | Multi-task | `bert-base-uncased` | LoRA | TweetEval (all 7 tasks) | Avg F1 |

## Datasets (Book + Extensions)

| Dataset | Task | Classes | Size | Book Ch |
|---------|------|---------|------|---------|
| `tweet_eval/sentiment` | Sentiment | 3 | 45K | Ch 2 |
| `tweet_eval/emotion` | Emotion | 4 | 45K | — |
| `tweet_eval/hate` | Hate detection | 2 | 45K | — |
| `glue/sst2` | Sentiment | 2 | 67K | — |
| `imdb` | Sentiment | 2 | 25K | — |
| `ag_news` | Topic | 4 | 120K | — |

## Quick Start

```bash
# Book baseline (full fine-tuning)
python -m phase1_classification.run \
  --config configs/phase1/bert-tweet-sentiment.yaml \
  --exp-id PH1-001

# With LoRA (see tracks/peft)
python -m phase1_classification.run \
  --config configs/phase1/bert-tweet-sentiment.yaml \
  --peft lora --peft-r 16 \
  --exp-id PH1-003

# Feature extraction baseline
python -m phase1_classification.run \
  --config configs/phase1/bert-tweet-sentiment.yaml \
  --freeze-backbone \
  --exp-id PH1-002
```

## Key Book Code Patterns to Reproduce

```python
# 1. Dataset exploration (Ch 2.1)
from datasets import load_dataset
ds = load_dataset("tweet_eval", "sentiment")
print(ds["train"][0])  # {'text': '...', 'label': 1}

# 2. Tokenization (Ch 2.2)
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True, max_length=128)
ds_tokenized = ds.map(tokenize, batched=True)

# 3. Feature extraction (Ch 2.3)
from transformers import AutoModel
import torch
model = AutoModel.from_pretrained("bert-base-uncased")
with torch.no_grad():
    outputs = model(**inputs)
cls_embeddings = outputs.last_hidden_state[:, 0]  # [CLS] token

# 4. Fine-tuning (Ch 2.4)
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
trainer = Trainer(model, args, train_dataset=ds_tokenized["train"], eval_dataset=ds_tokenized["validation"])
trainer.train()
```

## Evaluation Protocol

- **Primary**: Macro F1 (class-imbalanced)
- **Secondary**: Per-class F1, accuracy, confusion matrix
- **Slices**: By tweet length, language, hashtags
- **Baselines**: LogisticRegression on TF-IDF, frozen BERT + LR

## Next Phase Dependencies

- Phase 2 (Architecture): Understand BERT internals for better debugging
- Phase 3 (NER): Token classification builds on same tokenization
- Tracks/PEFT: Apply LoRA/QLoRA here first
- Tracks/Evaluation: Add GLUE benchmark suite