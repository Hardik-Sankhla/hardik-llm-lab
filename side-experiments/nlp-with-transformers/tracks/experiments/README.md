# Experiments Track — Unified Registry

> **Book Context**: All chapters — structured experiment tracking following main lab pattern  
> **Core**: Single source of truth for all runs across all phases and tracks

## Registry Structure

```
tracks/experiments/
├── configs/                 # Experiment YAML configs (source of truth)
│   ├── PH0-FDN-001.yaml
│   ├── PH1-CLS-001.yaml
│   ├── PH1-CLS-002.yaml
│   ├── PH3-NER-001.yaml
│   ├── PH4-GEN-001.yaml
│   ├── PH5-SUM-001.yaml
│   ├── PH6-QA-001.yaml
│   ├── PH7-EFF-001.yaml
│   ├── PH8-LR-001.yaml
│   ├── PH9-SCR-001.yaml
│   ├── PH10-FRO-001.yaml
│   ├── PEFT-PH1-001.yaml
│   ├── EVAL-PH1-001.yaml
│   └── DEP-PH7-001.yaml
├── runs/                    # Runtime artifacts (gitignored)
│   ├── PH1-CLS-001/
│   │   ├── checkpoints/
│   │   ├── logs/
│   │   ├── tensorboard/
│   │   └── wandb/
│   └── ...
├── results/                 # Aggregated metrics (committed)
│   ├── PH1-CLS-001.json
│   ├── PH3-NER-001.json
│   └── ...
├── plots/                   # Visualizations (committed)
│   ├── PH1-CLS-001/
│   │   ├── training_loss.png
│   │   ├── learning_rate.png
│   │   └── confusion_matrix.png
│   └── ...
└── registry.csv             # Master index of all experiments
```

## Experiment ID Format

```
{TRACK}-{PHASE}-{TASK}-{NNN}
```

| Track Prefix | Track | Example |
|--------------|-------|---------|
| `PH` | Phase (main book chapters) | `PH1-CLS-001`, `PH3-NER-001` |
| `PEFT` | PEFT track | `PEFT-PH1-001`, `PEFT-PH4-001` |
| `EVAL` | Evaluation track | `EVAL-PH1-001`, `EVAL-FULL-001` |
| `DEP` | Deployment track | `DEP-PH7-001`, `DEP-PH4-001` |
| `DST` | Distillation | `DST-PH7-001` |

## Phase Task Codes

| Code | Task | Phase |
|------|------|-------|
| `FDN` | Foundations | 0 |
| `CLS` | Classification | 1 |
| `ARCH` | Architecture | 2 |
| `NER` | NER & Multilingual | 3 |
| `GEN` | Generation | 4 |
| `SUM` | Summarization | 5 |
| `QA` | QA & RAG | 6 |
| `EFF` | Efficiency | 7 |
| `LR` | Low-Resource | 8 |
| `SCR` | From Scratch | 9 |
| `FRO` | Frontiers | 10 |

## Registry CSV Schema

```csv
exp_id,track,phase,task,model,peft,dataset,metric,value,status,started_at,completed_at,duration_s,config_hash,notes
PH1-CLS-001,PH,1,CLS,bert-base-uncased,none,tweet_eval/sentiment,f1,0.723,completed,2024-01-15T10:00:00Z,2024-01-15T10:45:00Z,2700,abc123,Book baseline
PH1-CLS-002,PH,1,CLS,bert-base-uncased,lora,tweet_eval/sentiment,f1,0.718,completed,2024-01-15T11:00:00Z,2024-01-15T11:30:00Z,1800,def456,LoRA r=16
PH3-NER-001,PH,3,NER,xlm-roberta-base,none,wikiann/en,f1,0.851,completed,2024-01-16T10:00:00Z,2024-01-16T11:20:00Z,4800,ghi789,Book baseline
PEFT-PH1-001,PEFT,1,CLS,bert-base-uncased,lora,tweet_eval/sentiment,f1,0.718,completed,2024-01-15T11:00:00Z,2024-01-15T11:30:00Z,1800,jkl012,LoRA r=16
EVAL-PH1-001,EVAL,1,CLS,bert-base-uncased,none,glue/sst2,accuracy,0.942,completed,2024-01-15T12:00:00Z,2024-01-15T12:10:00Z,600,mno345,Post-training eval
DEP-PH7-001,DEP,7,EFF,distilbert-base-uncased,none,clinc_oos,latency_p50,0.042,completed,2024-01-17T10:00:00Z,2024-01-17T10:05:00Z,300,pqr678,ONNX INT8
```

## Experiment Config Schema (YAML)

```yaml
# tracks/experiments/configs/PH1-CLS-001.yaml
experiment:
  id: "PH1-CLS-001"
  track: "PH"
  phase: 1
  task: "CLS"
  name: "BERT-base on TweetEval Sentiment (Book Baseline)"
  description: "Reproduce Ch 2 Text Classification with BERT fine-tuning"
  tags: ["book-baseline", "classification", "bert", "tweet-eval"]
  seed: 42

model:
  name: "bert-base-uncased"
  revision: "main"
  torch_dtype: "float32"

peft:
  method: "none"  # or "lora", "qlora", "adalora", "ia3"
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: ["query", "value", "key", "dense"]
  bias: "none"
  task_type: "SEQ_CLS"

data:
  dataset: "tweet_eval"
  subset: "sentiment"
  max_length: 128
  train_split: "train"
  eval_split: "validation"
  test_split: "test"
  num_proc: 4

training:
  output_dir: "tracks/experiments/runs/PH1-CLS-001"
  num_epochs: 3
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
  metric_for_best_model: "eval_f1"
  greater_is_better: true
  report_to: ["tensorboard", "wandb"]
  run_name: "PH1-CLS-001"

evaluation:
  metrics: ["accuracy", "f1_macro", "matthews"]
  per_class: true
  eval_on_test: true

deployment:
  export_onnx: false
  quantize: null
```

## CLI Commands

```bash
# List all experiments
python -m tracks.experiments.list

# List by phase
python -m tracks.experiments.list --phase 1

# List by track
python -m tracks.experiments.list --track PEFT

# Show experiment details
python -m tracks.experiments.show PH1-CLS-001

# Run experiment
python -m tracks.experiments.run --exp-id PH1-CLS-001

# Run with overrides
python -m tracks.experiments.run --exp-id PH1-CLS-001 --override training.batch_size=64

# Resume from checkpoint
python -m tracks.experiments.run --exp-id PH1-CLS-001 --resume tracks/experiments/runs/PH1-CLS-001/checkpoints/best.pt

# Compare experiments
python -m tracks.experiments.compare PH1-CLS-001 PH1-CLS-002 PEFT-PH1-001

# Generate report
python -m tracks.experiments.report --exp-id PH1-CLS-001 --output tracks/experiments/runs/PH1-CLS-001/report.md

# Sweep hyperparameters
python -m tracks.experiments.sweep --config tracks/experiments/configs/sweeps/lora-rank.yaml

# Clean up
python -m tracks.experiments.clean --status failed
python -m tracks.experiments.clean --older-than 30d
```

## Results JSON Schema

```json
{
  "exp_id": "PH1-CLS-001",
  "track": "PH",
  "phase": 1,
  "task": "CLS",
  "config_hash": "abc123def456",
  "status": "completed",
  "started_at": "2024-01-15T10:00:00Z",
  "completed_at": "2024-01-15T10:45:00Z",
  "duration_seconds": 2700,
  "hardware": {"gpu": "RTX 3090", "count": 1, "cuda_version": "12.1"},
  "config": { ...full resolved config... },
  "metrics": {
    "train_loss": [0.69, 0.42, 0.31, ...],
    "eval_f1_macro": [0.65, 0.70, 0.72, 0.723],
    "eval_accuracy": [0.68, 0.72, 0.74, 0.741],
    "eval_matthews": [0.58, 0.64, 0.68, 0.685],
    "best_eval_f1_macro": 0.723,
    "best_step": 1500
  },
  "final_checkpoint": "tracks/experiments/runs/PH1-CLS-001/checkpoints/best.pt",
  "artifacts": {
    "tensorboard": "tracks/experiments/runs/PH1-CLS-001/tensorboard",
    "wandb_run": "hardik-llm-lab/PH1-CLS-001/abc123",
    "model_card": "tracks/experiments/runs/PH1-CLS-001/model_card.md"
  }
}
```

## Integration with Main Lab

This registry follows the same pattern as [`../../../experiments/`](../../../experiments/) in the main LLM Lab. The `tracks.experiments.run` module shares the same interface — only the config schema differs (transformers vs. from-scratch).

## Quick Start

```bash
# Create new experiment from template
python -m tracks.experiments.create \
  --template classification-lora \
  --phase 1 \
  --model bert-base-uncased \
  --dataset tweet_eval/sentiment \
  --output tracks/experiments/configs/PH1-CLS-003.yaml

# Validate config
python -m tracks.experiments.validate tracks/experiments/configs/PH1-CLS-003.yaml

# Run and auto-log
python -m tracks.experiments.run --exp-id PH1-CLS-003 --log-to-registry
```