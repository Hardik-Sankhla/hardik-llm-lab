# Scripts — Phase-Aligned Utilities

> **Structure**: Organized by function, reusable across all phases and tracks

## Directory Structure

```
scripts/
├── data/                    # Data download, preprocessing, formatting
│   ├── download_tweet_eval.py       # Ch 2: TweetEval (Phase 1)
│   ├── download_wikiann.py          # Ch 4: WikiANN (Phase 3)
│   ├── download_cnn_dailymail.py    # Ch 6: CNN/DM (Phase 5)
│   ├── download_squad.py            # Ch 7: SQuAD (Phase 6)
│   ├── download_clinc_oos.py        # Ch 8: CLINC (Phase 7)
│   ├── download_github_issues.py    # Ch 9: GitHub Issues (Phase 8)
│   ├── download_codeparrot.py       # Ch 10: CodeParrot (Phase 9)
│   ├── format_alpaca.py             # Phase 4: Alpaca format
│   └── split_dataset.py
├── training/                # Training utilities
│   ├── trainer_wrapper.py           # Custom Trainer subclasses
│   ├── distillation_trainer.py      # Ch 8: DistillationTrainer
│   ├── callbacks.py                 # EarlyStopping, WandB, etc.
│   ├── schedulers.py                # Cosine, Linear, Constant
│   └── optimizers.py                # AdamW, Adafactor, Lion
├── evaluation/              # Evaluation helpers
│   ├── compute_metrics.py           # Phase-specific metric fns
│   ├── run_benchmark.py             # Benchmark runner
│   ├── compare_runs.py              # Compare experiment results
│   └── seqeval_metrics.py           # Ch 4: NER metrics
├── deployment/              # Export, quantize, serve
│   ├── export_onnx.py               # Ch 8: ONNX export
│   ├── quantize_int8.py             # Ch 8: Dynamic/Static INT8
│   ├── quantize_gptq.py             # Modern: GPTQ 4-bit
│   ├── quantize_awq.py              # Modern: AWQ 4-bit
│   ├── convert_gguf.py              # Modern: GGUF conversion
│   ├── serve_vllm.py                # Modern: vLLM server
│   ├── serve_tgi.py                 # Modern: TGI server
│   └── serve_onnx.py                # Ch 8: ONNX Runtime server
├── analysis/                # Model analysis
│   ├── attention_viz.py             # Ch 1,3: Attention visualization
│   ├── layer_probe.py               # Ch 3: Layer-wise probing
│   ├── parameter_count.py           # Ch 3: Param counting
│   ├── tokenizer_stats.py           # Ch 2,10: Fertility, compression
│   └── scaling_law_fit.py           # Ch 11: Kaplan/Chinchilla fit
├── utils/                   # Shared utilities
│   ├── seeding.py                   # set_seed(), deterministic
│   ├── logging.py                   # Structured logging, JSONL
│   ├── config.py                    # load_config(), merge_configs()
│   ├── device.py                    # get_device(), auto_batch_size()
│   ├── hub.py                       # push_to_hub(), create_repo()
│   └── distributed.py               # Accelerate, FSDP helpers
└── phase/                     # Phase entry points
    ├── run_phase0.py                # Ch 1: Foundations demo
    ├── run_phase1.py                # Ch 2: Classification
    ├── run_phase2.py                # Ch 3: Architecture
    ├── run_phase3.py                # Ch 4: NER
    ├── run_phase4.py                # Ch 5: Generation
    ├── run_phase5.py                # Ch 6: Summarization
    ├── run_phase6.py                # Ch 7: QA/RAG
    ├── run_phase7.py                # Ch 8: Efficiency
    ├── run_phase8.py                # Ch 9: Low-resource
    ├── run_phase9.py                # Ch 10: From Scratch
    └── run_phase10.py               # Ch 11: Frontiers
```

## Script Standards

| Requirement | Detail |
|-------------|--------|
| **CLI** | `argparse` with `--help`, typed args, subcommands |
| **Config** | Load YAML config (`--config`), override with CLI (`--override key=value`) |
| **Logging** | Structured logging (stdout + JSONL to `logs/`), `logging.INFO` default |
| **Seeding** | `set_seed(args.seed)` at entry (default 42) |
| **Device** | Auto-detect CUDA/MPS/CPU via `get_device()` |
| **Outputs** | Save to `tracks/experiments/runs/{phase}/{exp-id}/{script-name}/` |
| **Config Hash** | Compute SHA256 of resolved config for reproducibility |

## Phase Entry Points (Main CLI)

Each `scripts/phase/run_phase{X}.py` is the main entry for that phase:

```bash
# Phase 1: Classification (Ch 2)
python scripts/phase/run_phase1.py \
  --config configs/phase1_classification/bert-tweet-sentiment.yaml \
  --exp-id PH1-CLS-001 \
  --mode train

# Phase 3: NER (Ch 4)
python scripts/phase/run_phase3.py \
  --config configs/phase3_ner_multilingual/xlm-roberta-wikiann-en.yaml \
  --exp-id PH3-NER-001 \
  --mode train

# Phase 4: Generation (Ch 5)
python scripts/phase/run_phase4.py \
  --config configs/phase4_generation/gpt2-nucleus.yaml \
  --exp-id PH4-GEN-001 \
  --mode generate

# Phase 7: Efficiency (Ch 8)
python scripts/phase/run_phase7.py \
  --config configs/phase7_efficiency/distill-intent.yaml \
  --exp-id PH7-EFF-001 \
  --mode distill
```

## Key Scripts by Book Chapter

### Ch 1 (Phase 0) — Foundations
```bash
# Pipeline tour
python scripts/phase/run_phase0.py --demo pipelines

# Attention visualization
python scripts/phase/run_phase0.py --demo attention --model bert-base-uncased

# Hub interaction
python scripts/phase/run_phase0.py --demo hub --push my-model
```

### Ch 2 (Phase 1) — Classification
```bash
# Download TweetEval
python scripts/data/download_tweet_eval.py --output data/tweet_eval/

# Train baseline
python scripts/phase/run_phase1.py --config configs/phase1_classification/bert-tweet-sentiment.yaml --exp-id PH1-CLS-001

# Evaluate
python scripts/evaluation/run_benchmark.py --phase phase1_classification --model bert-base-uncased --exp-id EVAL-PH1-001
```

### Ch 3 (Phase 2) — Architecture
```bash
# Train tiny encoder
python scripts/phase/run_phase2.py --config configs/phase2_architecture/tiny-encoder.yaml --exp-id PH2-ARCH-001

# Attention visualization
python scripts/analysis/attention_viz.py --model bert-base-uncased --text "The cat sat"
```

### Ch 4 (Phase 3) — NER
```bash
# Download WikiANN
python scripts/data/download_wikiann.py --langs en,de,fr,es --output data/wikiann/

# Train XLM-RoBERTa
python scripts/phase/run_phase3.py --config configs/phase3_ner_multilingual/xlm-roberta-wikiann-en.yaml --exp-id PH3-NER-001
```

### Ch 5 (Phase 4) — Generation
```bash
# Decoding comparison
python scripts/phase/run_phase4.py --config configs/phase4_generation/gpt2-nucleus.yaml --exp-id PH4-GEN-001

# Temperature sweep
python scripts/phase/run_phase4.py --config configs/phase4_generation/gpt2-temp-sweep.yaml --exp-id PH4-GEN-002
```

### Ch 6 (Phase 5) — Summarization
```bash
# Download CNN/DM
python scripts/data/download_cnn_dailymail.py --output data/cnn_dailymail/

# Train PEGASUS
python scripts/phase/run_phase5.py --config configs/phase5_summarization/pegasus-cnn-dailymail.yaml --exp-id PH5-SUM-001
```

### Ch 7 (Phase 6) — QA/RAG
```bash
# Download SQuAD
python scripts/data/download_squad.py --output data/squad/

# Haystack pipeline
python scripts/phase/run_phase6.py --config configs/phase6_qa_rag/haystack-bm25-roberta.yaml --exp-id PH6-QA-001
```

### Ch 8 (Phase 7) — Efficiency
```bash
# Download CLINC
python scripts/data/download_clinc_oos.py --output data/clinc_oos/

# Distillation
python scripts/phase/run_phase7.py --config configs/phase7_efficiency/distill-intent.yaml --exp-id PH7-EFF-001

# Quantization
python scripts/deployment/quantize_int8.py --model bert-base-uncased --method static_int8 --calibration-data data/clinc_oos/
```

### Ch 9 (Phase 8) — Low-Resource
```bash
# Download GitHub Issues
python scripts/data/download_github_issues.py --output data/github_issues/

# Few-shot
python scripts/phase/run_phase8.py --config configs/phase8_low_resource/few-shot.yaml --k-shots 10 --exp-id PH8-LR-001

# Semi-supervised
python scripts/phase/run_phase8.py --config configs/phase8_low_resource/semi-supervised.yaml --exp-id PH8-LR-002
```

### Ch 10 (Phase 9) — From Scratch
```bash
# Download CodeParrot
python scripts/data/download_codeparrot.py --streaming --output data/codeparrot/

# Train BPE tokenizer
python scripts/phase/run_phase9.py --config configs/phase9_from_scratch/tokenizer.yaml --exp-id PH9-SCR-001

# Pretrain tiny GPT
python scripts/phase/run_phase9.py --config configs/phase9_from_scratch/tiny-gpt.yaml --exp-id PH9-SCR-002
```

### Ch 11 (Phase 10) — Frontiers
```bash
# FlashAttention
python scripts/phase/run_phase10.py --config configs/phase10_frontiers/flash-attention.yaml --exp-id PH10-FRO-001

# ViT
python scripts/phase/run_phase10.py --config configs/phase10_frontiers/vit.yaml --exp-id PH10-FRO-002

# CLIP
python scripts/phase/run_phase10.py --config configs/phase10_frontiers/clip.yaml --exp-id PH10-FRO-003
```

## Cross-Phase Utility Scripts

```bash
# Config management
python scripts/utils/config.py validate configs/phase1_classification/bert-tweet-sentiment.yaml
python scripts/utils/config.py generate --phase 1 --template classification-lora --output configs/phase1_classification/new.yaml

# Experiment registry
python scripts/utils/experiments.py list --phase 1
python scripts/utils/experiments.py compare PH1-CLS-001 PH1-CLS-002
python scripts/utils/experiments.py report PH1-CLS-001

# Deployment
python scripts/deployment/export_onnx.py --model bert-base-uncased --output-dir tracks/experiments/runs/PH7-EFF-001/onnx/
python scripts/deployment/quantize_gptq.py --model mistralai/Mistral-7B-v0.1 --bits 4 --group-size 128
python scripts/deployment/serve_vllm.py --model mistralai/Mistral-7B-Instruct-v0.2 --port 8000
```

## Adding a New Script

1. Create in appropriate subdirectory (`data/`, `training/`, `evaluation/`, `deployment/`, `analysis/`, `utils/`, `phase/`)
2. Follow standards: CLI, Config, Logging, Seeding, Device, Outputs
3. Add entry in this README
4. Test with `--help`
5. Add to phase entry point if needed