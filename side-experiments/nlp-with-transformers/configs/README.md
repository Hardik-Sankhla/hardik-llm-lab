# Configuration Files — Phase-Aligned

> **Structure**: Phase-specific YAML configs that compose base configs (model, task, PEFT, training)

## Directory Structure

```
configs/
├── phase0_foundations/      # Ch 1: Pipelines, Hub, Tokenizers
├── phase1_classification/   # Ch 2: BERT, DistilBERT, TweetEval
├── phase2_architecture/     # Ch 3: Encoder, Decoder, Configs
├── phase3_ner_multilingual/ # Ch 4: XLM-RoBERTa, WikiANN
├── phase4_generation/       # Ch 5: GPT-2, Decoding Strategies
├── phase5_summarization/    # Ch 6: PEGASUS, BART, T5, CNN/DM
├── phase6_qa_rag/           # Ch 7: Haystack, BM25/DPR, SQuAD
├── phase7_efficiency/       # Ch 8: Distillation, Quant, ONNX, Pruning
├── phase8_low_resource/     # Ch 9: Few-shot, Aug, Semi-sup
├── phase9_from_scratch/     # Ch 10: Tokenizer, Pretrain, Scaling
├── phase10_frontiers/       # Ch 11: FlashAttn, ViT, CLIP, Whisper
├── base/                    # Shared base configs
│   ├── model/
│   ├── task/
│   ├── peft/
│   ├── training/
│   └── evaluation/
└── tracks/                  # Cross-phase track configs
    ├── peft/
    ├── evaluation/
    └── deployment/
```

## Config Composition Pattern

Each phase config composes base configs + phase-specific overrides:

```yaml
# configs/phase1_classification/bert-tweet-sentiment.yaml
# Base configs (auto-loaded)
#   base/model/bert-base-uncased.yaml
#   base/task/classification.yaml
#   base/training/finetune.yaml
# Phase overrides (this file)
experiment:
  id: "PH1-CLS-001"
  phase: 1
  task: "CLS"
  name: "BERT-base on TweetEval Sentiment (Book Ch 2 Baseline)"
  tags: ["book-baseline", "classification", "bert", "tweet-eval"]

model:
  name: "bert-base-uncased"
  # revision, torch_dtype inherited from base/model/bert-base-uncased.yaml

data:
  dataset: "tweet_eval"
  subset: "sentiment"
  # max_length, splits inherited from base/task/classification.yaml

training:
  num_epochs: 3
  learning_rate: 2e-5
  # batch_size, scheduler inherited from base/training/finetune.yaml

evaluation:
  metrics: ["accuracy", "f1_macro", "matthews"]
  # per_class inherited
```

## Base Config Templates

### `base/model/bert-base-uncased.yaml`
```yaml
model:
  name: "bert-base-uncased"
  revision: "main"
  torch_dtype: "float32"
  num_labels: 3  # overridden per task
```

### `base/task/classification.yaml`
```yaml
data:
  max_length: 128
  train_split: "train"
  eval_split: "validation"
  test_split: "test"
  num_proc: 4

training:
  batch_size: 32
  gradient_accumulation: 1
  learning_rate: 2e-5
  weight_decay: 0.01
  warmup_ratio: 0.06
  lr_scheduler: "linear"
  optimizer: "adamw_torch"
  fp16: false
  bf16: true
  gradient_checkpointing: false
  logging_steps: 50
  eval_steps: 500
  save_steps: 500
  save_total_limit: 2
  load_best_model_at_end: true
  metric_for_best_model: "eval_f1_macro"
  greater_is_better: true
  report_to: ["tensorboard", "wandb"]

evaluation:
  metrics: ["accuracy", "f1_macro", "matthews"]
  per_class: true
  eval_on_test: true
```

### `base/peft/lora-default.yaml`
```yaml
peft:
  method: "lora"
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: ["query", "value", "key", "dense"]
  bias: "none"
  task_type: "SEQ_CLS"
```

## Phase-Specific Config Examples

### Phase 1 (Classification) — `configs/phase1_classification/`
| Config | Model | Dataset | PEFT | Book Section |
|--------|-------|---------|------|--------------|
| `bert-tweet-sentiment.yaml` | BERT-base | TweetEval sentiment | None | Ch 2.4 |
| `bert-tweet-sentiment-lora.yaml` | BERT-base | TweetEval sentiment | LoRA r=16 | Ch 9.4 |
| `distilbert-tweet-sentiment.yaml` | DistilBERT | TweetEval sentiment | None | Ch 8.4 |
| `bert-tweet-emotion.yaml` | BERT-base | TweetEval emotion | None | — |
| `bert-glue-sst2.yaml` | BERT-base | GLUE SST-2 | None | — |

### Phase 3 (NER) — `configs/phase3_ner_multilingual/`
| Config | Model | Dataset | PEFT | Book Section |
|--------|-------|---------|------|--------------|
| `xlm-roberta-wikiann-en.yaml` | XLM-R-base | WikiANN en | None | Ch 4.5 |
| `xlm-roberta-wikiann-multi.yaml` | XLM-R-base | WikiANN en,de,fr,es | None | Ch 4.7 |
| `xlm-roberta-wikiann-lora.yaml` | XLM-R-base | WikiANN multi | LoRA r=16 | — |
| `mbert-wikiann-en.yaml` | mBERT | WikiANN en | None | Ch 4.2 |

### Phase 4 (Generation) — `configs/phase4_generation/`
| Config | Model | Decoding | Book Section |
|--------|-------|----------|--------------|
| `gpt2-greedy.yaml` | GPT-2 | Greedy | Ch 5.2 |
| `gpt2-beam.yaml` | GPT-2 | Beam (5) | Ch 5.3 |
| `gpt2-topk.yaml` | GPT-2 | Top-k (50) | Ch 5.5 |
| `gpt2-nucleus.yaml` | GPT-2 | Top-p (0.9) | Ch 5.5 |
| `gpt2-temp-sweep.yaml` | GPT-2 | T=0.1..1.5 | Ch 5.4 |
| `gpt2-medium-nucleus.yaml` | GPT-2-med | Top-p (0.9) | — |

### Phase 7 (Efficiency) — `configs/phase7_efficiency/`
| Config | Technique | Teacher | Student | Book Section |
|--------|-----------|---------|---------|--------------|
| `intent-baseline.yaml` | Full FT | — | DistilBERT | Ch 8.1-8.2 |
| `distill-intent.yaml` | Distillation | BERT-base | DistilBERT | Ch 8.3-8.4 |
| `distill-intent-optuna.yaml` | Distill + HPO | BERT-base | DistilBERT | Ch 8.5 |
| `quantize-dynamic.yaml` | Dynamic INT8 | — | BERT-base | Ch 8.6 |
| `quantize-static.yaml` | Static INT8 | — | BERT-base | Ch 8.6 |
| `onnx-intent.yaml` | ONNX export | — | DistilBERT | Ch 8.7 |
| `prune-intent.yaml` | Pruning 50% | — | BERT-base | Ch 8.8 |

## Config Loading (Python)

```python
from scripts.utils.config import load_config, merge_configs

# Auto-discover and compose phase config
def load_phase_config(phase_dir: str, config_name: str):
    """Load base configs + phase config + track configs."""
    base_configs = [
        "base/model",
        "base/task",
        "base/training",
        "base/evaluation",
    ]

    configs = []
    for base in base_configs:
        path = f"configs/{base}/{config_name}.yaml"
        if os.path.exists(path):
            configs.append(load_config(path))

    # Phase-specific config
    phase_path = f"configs/{phase_dir}/{config_name}.yaml"
    configs.append(load_config(phase_path))

    # Track configs (optional)
    for track in ["peft", "evaluation", "deployment"]:
        track_path = f"configs/tracks/{track}/{config_name}.yaml"
        if os.path.exists(track_path):
            configs.append(load_config(track_path))

    return merge_configs(configs)
```

## Quick Start

```bash
# Validate phase config
python scripts/utils/config.py validate configs/phase1_classification/bert-tweet-sentiment.yaml

# Generate from template
python scripts/utils/config.py generate \
  --phase 1 \
  --template classification \
  --model bert-base-uncased \
  --dataset tweet_eval/sentiment \
  --peft lora \
  --output configs/phase1_classification/bert-tweet-sentiment-lora.yaml

# List available configs for a phase
ls configs/phase1_classification/
```

## Naming Convention

```
{model}-{dataset}-{variant}.yaml
```

| Variant | Meaning |
|---------|---------|
| (none) | Book baseline, full fine-tuning |
| `-lora` | LoRA fine-tuning |
| `-qlora` | QLoRA (4-bit) |
| `-distill` | Knowledge distillation |
| `-quant` | Quantized inference |
| `-multi` | Multi-lingual / multi-task |

## Integration

- Phase configs → `tracks/experiments/configs/` for registry
- Track configs → `configs/tracks/{peft,evaluation,deployment}/`
- Base configs → Single source of truth for defaults