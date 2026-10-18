# Phase 3: Multilingual NER — Chapter 4

> **Book**: Ch 4 "Multilingual Named Entity Recognition" (pp. 111–145)  
> **Core**: WikiANN NER with XLM-RoBERTa — multilingual tokenization, cross-lingual transfer, zero-shot

## Book Implementation Checklist

| Section | Book Task | Key Code | Extension Ideas |
|---------|-----------|----------|-----------------|
| 4.1 | WikiANN dataset (multi-lang) | `load_dataset("wikiann", "en")` | Try all 40+ languages |
| 4.2 | Multilingual transformers | `xlm-roberta-base` | Compare mBERT, XLM-R variants |
| 4.3 | Tokenization deep dive | `tokenizer.word_ids()` for alignment | Subword label propagation strategies |
| 4.4 | Token classification model | `AutoModelForTokenClassification` | Custom head, CRF layer (Ch 4.4) |
| 4.5 | Fine-tune XLM-RoBERTa | `Trainer` with NER metrics | LoRA for efficient multi-lang FT |
| 4.6 | Error analysis | Per-entity, per-language F1 | Confusion by entity type |
| 4.7 | Cross-lingual transfer | Train en → test de/fr/es | Zero-shot vs few-shot per lang |
| 4.8 | Multi-lingual training | Mix languages in batch | Language balancing strategies |

## Recommended Experiments

| Exp ID | Variant | Model | Languages | PEFT | Target |
|--------|---------|-------|-----------|------|--------|
| PH3-001 | Book baseline (en) | `xlm-roberta-base` | en only | Full FT | F1 ≥ 0.85 |
| PH3-002 | Zero-shot transfer | `xlm-roberta-base` | en→de,fr,es | Full FT | F1 ≥ 0.65 |
| PH3-003 | Multi-lingual FT | `xlm-roberta-base` | en,de,fr,es | Full FT | Avg F1 ≥ 0.78 |
| PH3-004 | LoRA multi-lang | `xlm-roberta-base` | en,de,fr,es | LoRA r=16 | Avg F1 ≥ 0.75 |
| PH3-005 | mBERT comparison | `bert-base-multilingual-cased` | en,de,fr,es | Full FT | Compare to XLM-R |
| PH3-006 | All 40 langs | `xlm-roberta-base` | WikiANN all | LoRA | Per-lang F1 table |

## Datasets

| Dataset | Languages | Entities | Size | Book Ch |
|---------|-----------|----------|------|---------|
| `wikiann` | 40+ | PER, ORG, LOC | ~20K/lang | Ch 4 |
| `conll2003` | en | PER, ORG, LOC, MISC | 14K | — |
| `masakhaner` | 10 African | PER, ORG, LOC, DATE | ~2K/lang | — |
| `panx` | 7 (XTREME) | PER, ORG, LOC | 20K/en | — |

## Quick Start

```bash
# Book baseline: English WikiANN
python -m phase3_ner_multilingual.run \
  --config configs/phase3/xlm-roberta-wikiann-en.yaml \
  --exp-id PH3-001

# Zero-shot cross-lingual (train en, eval de/fr/es)
python -m phase3_ner_multilingual.run \
  --config configs/phase3/xlm-roberta-wikiann-en.yaml \
  --eval-langs de,fr,es \
  --exp-id PH3-002

# Multi-lingual fine-tuning
python -m phase3_ner_multilingual.run \
  --config configs/phase3/xlm-roberta-wikiann-multi.yaml \
  --train-langs en,de,fr,es \
  --exp-id PH3-003

# With LoRA (tracks/peft)
python -m phase3_ner_multilingual.run \
  --config configs/phase3/xlm-roberta-wikiann-multi.yaml \
  --peft lora --peft-r 16 \
  --exp-id PH3-004
```

## Key Book Code Patterns to Reproduce

```python
# 1. WikiANN loading (Ch 4.1)
from datasets import load_dataset
# Single language
ds_en = load_dataset("wikiann", "en")
# Multiple languages
ds_multi = {lang: load_dataset("wikiann", lang) for lang in ["en", "de", "fr", "es"]}

# 2. Tokenization with alignment (Ch 4.3)
def tokenize_and_align_labels(examples):
    tokenized = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)  # or label[word_idx] for all subwords
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized["labels"] = labels
    return tokenized

# 3. XLM-RoBERTa for token classification (Ch 4.4-4.5)
from transformers import AutoModelForTokenClassification
model = AutoModelForTokenClassification.from_pretrained(
    "xlm-roberta-base", num_labels=7, id2label=id2label, label2id=label2id
)

# 4. Metrics (seqeval) (Ch 4.5)
from seqeval.metrics import f1_score, classification_report
predictions = np.argmax(logits, axis=2)
true_predictions = [[id2label[p] for (p, l) in zip(pred, label) if l != -100]
                    for pred, label in zip(predictions, labels)]
true_labels = [[id2label[l] for (p, l) in zip(pred, label) if l != -100]
               for pred, label in zip(predictions, labels)]
print(classification_report(true_labels, true_predictions))
```

## Cross-Lingual Transfer Strategies (Ch 4.7)

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| **Zero-shot** | Train source → test target | Target has no labels |
| **Few-shot** | Train source + k-shot target | Target has <100 labels |
| **Multi-source** | Train on all source langs | Multiple rich-resource langs |
| **Language-adaptive** | Continue MLM on target lang | Target lang very different |

## Evaluation Protocol

- **Primary**: Entity-level F1 (seqeval)
- **Per-language**: F1 for each language
- **Per-entity**: F1 for PER, ORG, LOC
- **Zero-shot gap**: Source F1 − Target F1

## Next Phase Dependencies

- Phase 4 (Generation): Decoder models for generative NER
- Phase 7 (Efficiency): Distill XLM-R for deployment
- Tracks/PEFT: LoRA critical for multi-lingual (prevents catastrophic forgetting)
- Tracks/Evaluation: XTREME benchmark suite