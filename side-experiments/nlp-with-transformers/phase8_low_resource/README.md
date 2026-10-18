# Phase 8: Low-Resource Learning — Chapter 9

> **Book**: Ch 9 "Dealing with Few to No Labels" (pp. 273–321)  
> **Core**: GitHub issue tagger — few-shot, data augmentation, embeddings lookup, in-context learning, semi-supervised

## Book Implementation Checklist

| Section | Book Task | Key Code | Extension Ideas |
|---------|-----------|----------|-----------------|
| 9.1 | GitHub issues dataset | `load_dataset("github-issues")` | Other low-resource datasets |
| 9.2 | Naive Bayes baseline | `sklearn.naive_bayes.MultinomialNB` | TF-IDF + classical ML |
| 9.3 | Zero-shot (no labels) | Embeddings + k-NN / clustering | Sentence-BERT, E5, BGE |
| 9.4 | Few-shot (k labels) | Fine-tune with k=10, 50, 100 | SetFit, ProtoNet, MAML |
| 9.5 | Data augmentation | Back-translation, EDA, LLM-generated | AugMix, CutMix for text |
| 9.6 | Embeddings as lookup | `sentence-transformers` + FAISS | RAG-style retrieval |
| 9.7 | In-context learning | Prompt + few-shot examples | Flan-T5, Llama-2, Mistral |
| 9.8 | Semi-supervised | Pseudo-labeling, consistency reg. | FixMatch, UDA, Mean Teacher |
| 9.9 | LM fine-tuning | Continue MLM on domain data | Domain-adaptive pretraining |

## Recommended Experiments

| Exp ID | Variant | Labels | Method | Dataset | Target F1 |
|--------|---------|--------|--------|---------|-----------|
| PH8-001 | Bayes baseline | Full | TF-IDF + NB | GitHub issues | ≥ 0.60 |
| PH8-002 | Zero-shot | 0 | Embeddings + k-NN | GitHub issues | ≥ 0.45 |
| PH8-003 | Few-shot k=10 | 10/class | Fine-tune | GitHub issues | ≥ 0.55 |
| PH8-004 | Few-shot k=50 | 50/class | Fine-tune | GitHub issues | ≥ 0.65 |
| PH8-005 | SetFit | 10/class | SetFit (SBERT) | GitHub issues | ≥ 0.70 |
| PH8-006 | Data aug + FT | 10/class | EDA + FT | GitHub issues | +5 F1 |
| PH8-007 | In-context | 10/class | Flan-T5 prompt | GitHub issues | ≥ 0.50 |
| PH8-008 | Semi-supervised | 10 labeled + unlabeled | Pseudo-label | GitHub issues | +8 F1 |
| PH8-009 | Domain LM FT | Unlabeled | Continue MLM | GitHub issues | +3 F1 |

## Quick Start

```bash
# Book: Naive Bayes baseline (Ch 9.2)
python -m phase8_low_resource.run \
  --config configs/phase8/naive-bayes.yaml \
  --exp-id PH8-001

# Book: Zero-shot with embeddings (Ch 9.3)
python -m phase8_low_resource.run \
  --config configs/phase8/zero-shot.yaml \
  --exp-id PH8-002

# Book: Few-shot fine-tuning (Ch 9.4)
python -m phase8_low_resource.run \
  --config configs/phase8/few-shot.yaml \
  --k-shots 10 \
  --exp-id PH8-003

# Book: Data augmentation (Ch 9.5)
python -m phase8_low_resource.run \
  --config configs/phase8/augmentation.yaml \
  --method eda \
  --exp-id PH8-006

# Book: In-context learning (Ch 9.7)
python -m phase8_low_resource.run \
  --config configs/phase8/in-context.yaml \
  --model google/flan-t5-base \
  --exp-id PH8-007

# Book: Semi-supervised (Ch 9.8-9.9)
python -m phase8_low_resource.run \
  --config configs/phase8/semi-supervised.yaml \
  --method pseudo-label \
  --exp-id PH8-008
```

## Key Book Code Patterns to Reproduce

```python
# 1. GitHub issues dataset (Ch 9.1)
from datasets import load_dataset
ds = load_dataset("github-issues", split="train")
# Filter for specific repos, extract title+body → label

# 2. Naive Bayes baseline (Ch 9.2)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
    ("clf", MultinomialNB())
])
pipeline.fit(train_texts, train_labels)
preds = pipeline.predict(test_texts)

# 3. Zero-shot with embeddings (Ch 9.3)
from sentence_transformers import SentenceTransformer
import faiss

encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
train_embs = encoder.encode(train_texts)
index = faiss.IndexFlatIP(train_embs.shape[1])
index.add(train_embs)

# For test: retrieve nearest train label
test_embs = encoder.encode(test_texts)
_, indices = index.search(test_embs, k=1)
preds = [train_labels[i[0]] for i in indices]

# 4. Few-shot fine-tuning (Ch 9.4)
# Subsample k examples per class
few_shot_ds = ds.shuffle(seed=42).select(range(k * num_classes))
trainer = Trainer(model, args, train_dataset=few_shot_ds, ...)

# 5. Data augmentation (Ch 9.5)
import nlpaug.augmenter.word as naw
aug = naw.BackTranslationAug(from_model_name="facebook/wmt19-en-de", to_model_name="facebook/wmt19-de-en")
augmented = aug.augment(text)

# EDA (Easy Data Augmentation)
def eda(text, alpha=0.1, n_aug=4):
    # Synonym replacement, random insertion, swap, deletion
    pass

# 6. In-context learning (Ch 9.7)
prompt = """Classify the GitHub issue:
Issue: "Login page throws 500 error"
Label: bug
Issue: "Add dark mode support"
Label: feature
Issue: "Typo in README"
Label: documentation
Issue: "{test_issue}"
Label:"""
output = flan_t5.generate(prompt)

# 7. Semi-supervised pseudo-labeling (Ch 9.8)
# 1. Train on labeled
# 2. Predict on unlabeled (high confidence)
# 3. Add pseudo-labeled to train set
# 4. Retrain
```

## Low-Resource Methods Comparison

| Method | Labels Needed | Complexity | Quality | Book Ch |
|--------|---------------|------------|---------|---------|
| **Classical ML (TF-IDF + NB)** | 0 (unsupervised) | Low | Baseline | 9.2 |
| **Embeddings + k-NN** | 0 | Low | Medium | 9.3 |
| **Prompting (in-context)** | 0-32 | Low | Variable | 9.7 |
| **SetFit** | 10-50/class | Medium | High | — |
| **Fine-tuning** | 10-100/class | Medium | High | 9.4 |
| **Data Aug + FT** | 10-50/class | Medium | High | 9.5 |
| **Semi-supervised** | 10-50 + unlabeled | High | Highest | 9.8-9.9 |

## Evaluation Protocol

| Metric | Purpose |
|--------|---------|
| **Few-shot F1** | Performance at k=10, 50, 100 |
| **Label efficiency** | F1 vs log(labels) curve |
| **Zero-shot gap** | Zero-shot vs full-supervised |
| **Augmentation gain** | Augmented vs non-augmented |
| **Pseudo-label precision** | Quality of generated labels |

## Modern Extensions (Beyond Book)

| Technique | Library | Use Case |
|-----------|---------|----------|
| **SetFit** | `setfit` | Few-shot with SBERT |
| **LLM data generation** | `llama`, `mistral` | Synthetic training data |
| **Prompt tuning** | `peft` | Soft prompts for few-shot |
| **Instruction tuning** | `trl` | Flan-style multi-task |
| **Active learning** | `modAL`, `small-text` | Select most informative samples |

## Next Phase Dependencies

- Phase 7 (Efficiency): Distill few-shot models for deployment
- Phase 9 (From Scratch): Domain-adaptive pretraining
- Tracks/PEFT: LoRA/IA³ critical for few-shot
- Tracks/Evaluation: Few-shot benchmarks (FewGLUE, SetFit benchmarks)