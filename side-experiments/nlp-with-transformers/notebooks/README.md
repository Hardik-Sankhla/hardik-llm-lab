# Notebooks — Phase-Aligned Exploration

> **Structure**: Phase-specific notebooks for exploration, prototyping, and analysis — each linked to book chapter

## Directory Structure (Phase-Aligned)

```
notebooks/
├── phase0_foundations/           # Ch 1: Hello Transformers
│   ├── 01-pipeline-tour.ipynb
│   ├── 02-attention-viz.ipynb
│   ├── 03-transfer-learning-demo.ipynb
│   ├── 04-hf-ecosystem.ipynb
│   └── 05-challenges-discussion.ipynb
├── phase1_classification/        # Ch 2: Text Classification
│   ├── 01-tweet-eval-exploration.ipynb
│   ├── 02-tokenization-comparison.ipynb
│   ├── 03-feature-extraction-vs-finetune.ipynb
│   ├── 04-error-analysis.ipynb
│   └── 05-multi-task-tweet-eval.ipynb
├── phase2_architecture/          # Ch 3: Transformer Anatomy
│   ├── 01-encoder-decoder-comparison.ipynb
│   ├── 02-self-attention-math.ipynb
│   ├── 03-positional-embeddings.ipynb
│   ├── 04-model-variants-tree.ipynb
│   └── 05-tiny-transformer-training.ipynb
├── phase3_ner_multilingual/      # Ch 4: Multilingual NER
│   ├── 01-wikiann-exploration.ipynb
│   ├── 02-multilingual-tokenization.ipynb
│   ├── 03-ner-model-anatomy.ipynb
│   ├── 04-cross-lingual-transfer.ipynb
│   ├── 05-error-analysis-per-lang.ipynb
│   └── 06-model-widgets.ipynb
├── phase4_generation/            # Ch 5: Text Generation
│   ├── 01-decoding-strategies.ipynb
│   ├── 02-temperature-sweep.ipynb
│   ├── 03-topk-vs-nucleus.ipynb
│   ├── 04-diversity-metrics.ipynb
│   └── 05-human-eval-setup.ipynb
├── phase5_summarization/         # Ch 6: Summarization
│   ├── 01-cnn-dailymail-exploration.ipynb
│   ├── 02-model-comparison-bart-pegasus-t5.ipynb
│   ├── 03-rouge-bleu-deep-dive.ipynb
│   ├── 04-pegasus-finetuning.ipynb
│   ├── 05-samsum-dialogue.ipynb
│   └── 06-controllable-summarization.ipynb
├── phase6_qa_rag/                # Ch 7: Question Answering
│   ├── 01-amazon-reviews-qa.ipynb
│   ├── 02-haystack-pipeline.ipynb
│   ├── 03-retriever-evaluation.ipynb
│   ├── 04-reader-evaluation.ipynb
│   ├── 05-domain-adaptation.ipynb
│   ├── 06-generative-qa.ipynb
│   └── 07-rag-pipeline.ipynb
├── phase7_efficiency/            # Ch 8: Production Efficiency
│   ├── 01-intent-detection-baseline.ipynb
│   ├── 02-distillation-analysis.ipynb
│   ├── 03-optuna-hpo.ipynb
│   ├── 04-quantization-comparison.ipynb
│   ├── 05-onnx-export.ipynb
│   ├── 06-pruning-analysis.ipynb
│   └── 07-benchmarking.ipynb
├── phase8_low_resource/          # Ch 9: Few/No Labels
│   ├── 01-github-issues-exploration.ipynb
│   ├── 02-naive-bayes-baseline.ipynb
│   ├── 03-zero-shot-embeddings.ipynb
│   ├── 04-few-shot-finetuning.ipynb
│   ├── 05-data-augmentation.ipynb
│   ├── 06-in-context-learning.ipynb
│   ├── 07-semi-supervised.ipynb
│   └── 08-label-efficiency-curves.ipynb
├── phase9_from_scratch/          # Ch 10: Training from Scratch
│   ├── 01-codeparrot-exploration.ipynb
│   ├── 02-bpe-tokenizer-training.ipynb
│   ├── 03-tokenizer-evaluation.ipynb
│   ├── 04-pretraining-objectives.ipynb
│   ├── 05-training-loop-debug.ipynb
│   ├── 06-scaling-law-fit.ipynb
│   └── 07-downstream-evaluation.ipynb
├── phase10_frontiers/            # Ch 11: Future Directions
│   ├── 01-scaling-laws.ipynb
│   ├── 02-flash-attention.ipynb
│   ├── 03-long-context.ipynb
│   ├── 04-vision-transformers.ipynb
│   ├── 05-clip-zero-shot.ipynb
│   ├── 06-whisper-asr.ipynb
│   ├── 07-blip2-vqa.ipynb
│   └── 08-research-directions.ipynb
├── tracks/                       # Cross-phase track notebooks
│   ├── peft/
│   │   ├── 01-lora-rank-sweep.ipynb
│   │   ├── 02-qlora-vs-lora.ipynb
│   │   ├── 03-target-modules.ipynb
│   │   └── 04-peft-comparison.ipynb
│   ├── evaluation/
│   │   ├── 01-glue-benchmark.ipynb
│   │   ├── 02-mmlu-eval.ipynb
│   │   └── 03-generation-metrics.ipynb
│   └── deployment/
│       ├── 01-latency-profiling.ipynb
│       ├── 02-quantization-impact.ipynb
│       └── 03-serving-comparison.ipynb
└── 99-scratch/                   # Throwaway experiments
```

## Notebook Standards

| Requirement | Detail |
|-------------|--------|
| **Metadata** | Title, author, date, tags, hardware (GPU), runtime estimate |
| **Environment** | `%pip install` cells at top (reproducible) |
| **Seeding** | `set_seed(42)` in first code cell |
| **Outputs** | Clear, saved figures/tables (not just inline) — use `plt.savefig()` |
| **Cleanup** | No giant outputs committed (use `.gitignore` or `nbstripout`) |
| **Links** | Reference config YAML, script, experiment ID in markdown cells |
| **Phase Tag** | First markdown cell: `# Phase X: Chapter Y — {Topic}` |

## Naming Convention

```
{XX}-{short-description}.ipynb
```

| Prefix | Phase | Example |
|--------|-------|---------|
| `01-05` | Phase 0 | `01-pipeline-tour.ipynb` |
| `01-05` | Phase 1 | `01-tweet-eval-exploration.ipynb` |
| `01-05` | Phase 2 | `01-encoder-decoder-comparison.ipynb` |
| `01-06` | Phase 3 | `01-wikiann-exploration.ipynb` |
| `01-05` | Phase 4 | `01-decoding-strategies.ipynb` |
| `01-06` | Phase 5 | `01-cnn-dailymail-exploration.ipynb` |
| `01-07` | Phase 6 | `01-amazon-reviews-qa.ipynb` |
| `01-07` | Phase 7 | `01-intent-detection-baseline.ipynb` |
| `01-08` | Phase 8 | `01-github-issues-exploration.ipynb` |
| `01-07` | Phase 9 | `01-codeparrot-exploration.ipynb` |
| `01-08` | Phase 10 | `01-scaling-laws.ipynb` |

## Quick Start

```bash
# Install notebook dependencies
pip install -e ".[notebooks]"

# Or manually
pip install jupyterlab ipywidgets matplotlib seaborn plotly nltk seqeval

# Run Jupyter
jupyter lab notebooks/

# Strip outputs before commit
pip install nbstripout && nbstripout --install
```

## Book Chapter → Notebook Mapping

| Book Ch | Phase | Key Notebooks | Purpose |
|---------|-------|---------------|---------|
| 1 | 0 | `01-pipeline-tour`, `02-attention-viz` | Reproduce Ch 1 figures |
| 2 | 1 | `01-tweet-eval-exploration`, `03-feature-extraction-vs-finetune` | Reproduce Ch 2 results |
| 3 | 2 | `02-self-attention-math`, `05-tiny-transformer-training` | Understand Ch 3 math |
| 4 | 3 | `01-wikiann-exploration`, `04-cross-lingual-transfer` | Reproduce Ch 4 NER |
| 5 | 4 | `01-decoding-strategies`, `03-topk-vs-nucleus` | Reproduce Ch 5 decoding |
| 6 | 5 | `02-model-comparison`, `04-pegasus-finetuning` | Reproduce Ch 6 summarization |
| 7 | 6 | `02-haystack-pipeline`, `05-domain-adaptation` | Reproduce Ch 7 QA |
| 8 | 7 | `02-distillation-analysis`, `04-quantization-comparison` | Reproduce Ch 8 efficiency |
| 9 | 8 | `03-zero-shot-embeddings`, `06-in-context-learning` | Reproduce Ch 9 low-resource |
| 10 | 9 | `02-bpe-tokenizer-training`, `06-scaling-law-fit` | Reproduce Ch 10 from-scratch |
| 11 | 10 | `02-flash-attention`, `04-vision-transformers` | Explore Ch 11 frontiers |

## Notebook Template

```markdown
# Phase 1: Chapter 2 — Text Classification
## 03-feature-extraction-vs-finetune.ipynb

**Author**: Hardik Sankhla  
**Date**: 2024-01-15  
**Tags**: classification, bert, feature-extraction, fine-tuning  
**Hardware**: RTX 3090 (24GB)  
**Runtime**: ~30 min  
**Book Ref**: Ch 2.3-2.4 (pp. 62-78)  
**Experiment**: PH1-CLS-001, PH1-CLS-002  
**Config**: `configs/phase1_classification/bert-tweet-sentiment.yaml`

---

```python
# %pip install transformers datasets evaluate accelerate -q
```

```python
# Imports & seeding
import torch, numpy as np
from transformers import set_seed
set_seed(42)
```

```python
# Load data (Ch 2.1)
from datasets import load_dataset
ds = load_dataset("tweet_eval", "sentiment")
```

```python
# Feature extraction (Ch 2.3)
# ... code from book ...
```

```python
# Fine-tuning (Ch 2.4)
# ... code from book ...
```

```python
# Comparison & visualization
# ... plots saved to notebooks/phase1_classification/plots/ ...
```
```

## Useful Extensions

| Extension | Purpose |
|-----------|---------|
| `jupyterlab-git` | Git integration in Jupyter |
| `jupyterlab-tensorboard` | TensorBoard in Jupyter |
| `ipywidgets` | Interactive widgets |
| `jupyter-resource-usage` | Memory/CPU monitor |
| `nbstripout` | Strip outputs on commit |
| `jupytext` | Sync .ipynb ↔ .py |

## Integration with Scripts & Experiments

- Notebooks → prototype → graduate to `scripts/phase/run_phase{X}.py`
- Notebook results → log to `tracks/experiments/results/`
- Notebook figures → save to `notebooks/{phase}/plots/` and reference in experiment reports
- Configs → reference `configs/{phase}/*.yaml` in notebook markdown