# Evaluation Track — Benchmarks & Metrics

> **Book Context**: Ch 2 (Classification metrics), Ch 4 (NER seqeval), Ch 5 (Perplexity), Ch 6 (ROUGE/BLEU), Ch 7 (QA EM/F1, Retriever recall), Ch 8 (Latency/throughput)  
> **Core**: Standardized evaluation across all phases — run after each phase completion

## Book Chapter → Evaluation Mapping

| Book Ch | Phase | Task | Primary Metrics | Book Section |
|---------|-------|------|-----------------|--------------|
| Ch 2 | 1 | Classification | Accuracy, F1, Matthews | 2.4 |
| Ch 4 | 3 | NER | Entity F1 (seqeval) | 4.5 |
| Ch 5 | 4 | Generation | Perplexity, Distinct-n | 5.1-5.6 |
| Ch 6 | 5 | Summarization | ROUGE-1/2/L, BLEU | 6.4 |
| Ch 7 | 6 | QA | EM, F1, Recall@k, MRR | 7.4-7.5 |
| Ch 8 | 7 | Efficiency | Latency, Throughput, Memory | 8.2 |
| Ch 9 | 8 | Low-resource | Few-shot F1, Label efficiency | 9.2-9.9 |
| Ch 10 | 9 | Pretraining | Train loss, Eval PPL, Downstream | 10.11 |
| Ch 11 | 10 | Frontiers | Task-specific | — |

## Benchmark Suites by Phase

| Phase | Benchmarks | Datasets | Metrics |
|-------|------------|----------|---------|
| **0** | Pipeline sanity | — | Output quality check |
| **1** | GLUE (subset) | SST-2, MRPC, QQP | Acc, F1, Matthews |
| **2** | Architecture sanity | WikiText-2 | PPL, convergence |
| **3** | XTREME NER | WikiANN (en, de, fr, es) | Entity F1 per lang |
| **4** | Generation quality | WikiText-103, custom | PPL, Distinct-1/2, Repetition |
| **5** | Summarization | CNN/DM, XSum, SAMSum | ROUGE-1/2/L |
| **6** | QA | SQuAD 2.0, NQ, TriviaQA | EM, F1, Recall@k |
| **7** | Efficiency | Intent (CLINC) | Latency p50/p99, Throughput |
| **8** | Few-shot | FewGLUE, SetFit benchmarks | F1@k shots |
| **9** | Scaling | Custom checkpoints | Loss curves, Scaling law fit |
| **10** | Frontier | LongBench, ImageNet, LibriSpeech | Task-specific |

## Evaluation Harness

```python
# tracks/evaluation/harness.py
from dataclasses import dataclass
from typing import List, Dict, Any
import json
from datetime import datetime

@dataclass
class EvalConfig:
    model: str
    phase: str
    tasks: List[str]
    metrics: List[str]
    batch_size: int = 8
    device: str = "auto"

class Evaluator:
    def __init__(self, config: EvalConfig):
        self.config = config
        self.results = {}

    def run(self) -> Dict[str, Any]:
        """Run all configured evaluations."""
        for task in self.config.tasks:
            self.results[task] = self._eval_task(task)
        return self.results

    def _eval_task(self, task: str) -> Dict[str, float]:
        # Dispatch to task-specific evaluator
        evaluators = {
            "classification": self._eval_classification,
            "ner": self._eval_ner,
            "generation": self._eval_generation,
            "summarization": self._eval_summarization,
            "qa": self._eval_qa,
            "efficiency": self._eval_efficiency,
        }
        return evaluators[task]()

    def save_results(self, path: str):
        output = {
            "model": self.config.model,
            "phase": self.config.phase,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "config_hash": self._hash_config(),
            "results": self.results,
        }
        with open(path, "w") as f:
            json.dump(output, f, indent=2)
```

## Quick Start

```bash
# Phase 1: Classification eval
python -m tracks.evaluation.run \
  --phase phase1_classification \
  --model bert-base-uncased \
  --tasks sst2,mrpc,qqp \
  --exp-id EVAL-PH1-001

# Phase 3: NER eval (multi-lang)
python -m tracks.evaluation.run \
  --phase phase3_ner_multilingual \
  --model xlm-roberta-base \
  --tasks wikiann_en,wikiann_de,wikiann_fr \
  --exp-id EVAL-PH3-001

# Phase 4: Generation eval
python -m tracks.evaluation.run \
  --phase phase4_generation \
  --model gpt2 \
  --tasks perplexity,distinct \
  --exp-id EVAL-PH4-001

# Phase 5: Summarization eval
python -m tracks.evaluation.run \
  --phase phase5_summarization \
  --model facebook/bart-large-cnn \
  --tasks cnn_dailymail,xsum,samsum \
  --exp-id EVAL-PH5-001

# Phase 6: QA eval
python -m tracks.evaluation.run \
  --phase phase6_qa_rag \
  --model deepset/roberta-base-squad2 \
  --tasks squad_v2,natural_questions \
  --exp-id EVAL-PH6-001

# Phase 7: Efficiency eval
python -m tracks.evaluation.run \
  --phase phase7_efficiency \
  --model distilbert-base-uncased \
  --tasks latency,throughput,memory \
  --exp-id EVAL-PH7-001

# Full benchmark suite
python -m tracks.evaluation.run \
  --phase all \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --exp-id EVAL-FULL-001
```

## Metric Implementations (Book-Aligned)

### Classification (Ch 2.4)
```python
from evaluate import load
accuracy = load("accuracy")
f1 = load("f1")
matthews = load("matthews_correlation")

def compute_cls_metrics(preds, labels):
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "f1_macro": f1.compute(predictions=preds, references=labels, average="macro")["f1"],
        "matthews": matthews.compute(predictions=preds, references=labels)["matthews_correlation"],
    }
```

### NER (Ch 4.5) — seqeval
```python
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report

def compute_ner_metrics(preds, labels, id2label):
    true_preds = [[id2label[p] for p, l in zip(pred, label) if l != -100] for pred, label in zip(preds, labels)]
    true_labels = [[id2label[l] for p, l in zip(pred, label) if l != -100] for pred, label in zip(preds, labels)]
    return {
        "entity_f1": f1_score(true_labels, true_preds),
        "entity_precision": precision_score(true_labels, true_preds),
        "entity_recall": recall_score(true_labels, true_preds),
        "per_class": classification_report(true_labels, true_preds, output_dict=True),
    }
```

### Generation (Ch 5)
```python
import math
from evaluate import load

perplexity = load("perplexity")

def compute_gen_metrics(model, tokenizer, texts):
    # Perplexity
    ppl = perplexity.compute(model_id=model, texts=texts)["mean_perplexity"]

    # Distinct-n
    def distinct_n(texts, n):
        ngrams, total = set(), 0
        for t in texts:
            tokens = t.split()
            for i in range(len(tokens) - n + 1):
                ngrams.add(tuple(tokens[i:i+n]))
                total += 1
        return len(ngrams) / total if total else 0

    return {"perplexity": ppl, "distinct_1": distinct_n(texts, 1), "distinct_2": distinct_n(texts, 2)}
```

### Summarization (Ch 6.4) — ROUGE
```python
from evaluate import load
import nltk

rouge = load("rouge")

def compute_summarization_metrics(preds, labels):
    # ROUGE expects newline-separated sentences
    decoded_preds = ["\n".join(nltk.sent_tokenize(p.strip())) for p in preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(l.strip())) for l in labels]
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    return {k: round(v * 100, 4) for k, v in result.items()}
```

### QA (Ch 7.4-7.5) — EM/F1 + Retriever
```python
from evaluate import load
squad = load("squad_v2")

def compute_qa_metrics(preds, labels):
    formatted_preds = [{"id": ex["id"], "prediction_text": p, "no_answer_probability": 0.0} for ex, p in zip(labels, preds)]
    formatted_labels = [{"id": ex["id"], "answers": ex["answers"]} for ex in labels]
    return squad.compute(predictions=formatted_preds, references=formatted_labels)

# Retriever metrics (Haystack style)
def compute_retriever_metrics(retrieved_docs, relevant_docs, k=10):
    recall = len(set(retrieved_docs[:k]) & set(relevant_docs)) / len(relevant_docs)
    mrr = 0
    for i, doc in enumerate(retrieved_docs[:k]):
        if doc in relevant_docs:
            mrr = 1 / (i + 1)
            break
    return {"recall@k": recall, "mrr@k": mrr}
```

### Efficiency (Ch 8.2)
```python
from transformers.utils import benchmark
import torch

def compute_efficiency_metrics(model, tokenizer, batch_sizes=[1, 8, 32], seq_len=128):
    results = {}
    for bs in batch_sizes:
        dummy = tokenizer(["hello world"] * bs, padding=True, max_length=seq_len, truncation=True, return_tensors="pt")
        bench = benchmark(model, dummy, n_iterations=100, warmup=10)
        results[f"batch_{bs}"] = {
            "latency_ms_mean": bench.latency.mean * 1000,
            "latency_ms_p95": bench.latency.p95 * 1000,
            "throughput_samples_s": bench.throughput.mean,
        }
    # Memory
    if torch.cuda.is_available():
        results["peak_memory_gb"] = torch.cuda.max_memory_allocated() / 1e9
    return results
```

## Results Format (Unified)

```json
{
  "model": "bert-base-uncased",
  "phase": "phase1_classification",
  "exp_id": "EVAL-PH1-001",
  "timestamp": "2024-01-15T10:30:00Z",
  "config_hash": "abc123",
  "results": {
    "sst2": {"accuracy": 0.942, "f1_macro": 0.940, "matthews": 0.881},
    "mrpc": {"accuracy": 0.891, "f1_macro": 0.912},
    "qqp": {"accuracy": 0.912, "f1_macro": 0.889}
  }
}
```

## Integration with Experiment Registry

```bash
# Auto-log to tracks/experiments
python -m tracks.evaluation.run \
  --phase phase1_classification \
  --model bert-base-uncased \
  --log-to-registry \
  --exp-id PH1-001
```

## Next Steps

- Add `lm-eval-harness` for MMLU, BigBench, HELM
- Add `RAGAS` for RAG evaluation (Phase 6+)
- Add `AlpacaEval` / `MT-Bench` for chat (Phase 4+)
- CI integration: fail if regression > 1% on core metrics