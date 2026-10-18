# NLP with Transformers — Guided Learning Path (Solo Leveling)

> **Source**: *Natural Language Processing with Transformers (Revised Edition)* — Tunstall, von Werra, Wolf (Hugging Face)  
> **Goal**: Reproduce every book chapter hands-on, then extend with modern practices (PEFT, evaluation, deployment)

---

## How to Use This Repo

| Mode | Description |
|------|-------------|
| **Guided** | Follow phases 0→10 sequentially, each phase = one book chapter |
| **Solo Leveling** | Pick a phase, run the implementation checklist, verify against book outputs |
| **Reference** | Use phase READMEs as cheat sheets for book code patterns |

---

## Phase Map (11 Chapters → 11 Phases)

| Phase | Chapter | Title | Book Pages | Core Implementation | Status |
|-------|---------|-------|------------|---------------------|--------|
| **0** | Ch 1 | Hello Transformers | 25–44 | Pipelines, HF Hub, tokenizers, transfer learning | 🟢 Ready |
| **1** | Ch 2 | Text Classification | 45–78 | TweetEval sentiment, BERT fine-tuning, feature extraction | 🟢 Ready |
| **2** | Ch 3 | Transformer Anatomy | 81–108 | Encoder/decoder from scratch, self-attention, positional embeddings | 🟢 Ready |
| **3** | Ch 4 | Multilingual NER | 111–145 | WikiANN, XLM-RoBERTa, cross-lingual transfer, zero-shot | 🟢 Ready |
| **4** | Ch 5 | Text Generation | 147–164 | GPT-2, greedy/beam/top-k/nucleus decoding | 🟢 Ready |
| **5** | Ch 6 | Summarization | 165–187 | CNN/DailyMail, PEGASUS/BART/T5, ROUGE/BLEU | 🟢 Ready |
| **6** | Ch 7 | Question Answering | 189–231 | Haystack retriever-reader, BM25/DPR, domain adaptation | 🟢 Ready |
| **7** | Ch 8 | Production Efficiency | 233–272 | Distillation, quantization, ONNX, pruning, benchmarking | 🟢 Ready |
| **8** | Ch 9 | Few/No Labels | 273–321 | GitHub issues, few-shot, data aug, in-context, semi-supervised | 🟢 Ready |
| **9** | Ch 10 | Training from Scratch | 323–367 | CodeParrot, BPE tokenizer, pretraining loop, scaling laws | 🟢 Ready |
| **10** | Ch 11 | Future Directions | 369–394 | FlashAttention, ViT, CLIP, Whisper, scaling laws | 🟢 Ready |

---

## Quick Start (Per Phase)

```bash
# 0. Setup
cd /mnt/Storage/hardik-llm-lab/side-experiments/nlp-with-transformers
pip install -e ".[dev]"  # or: pip install transformers datasets accelerate evaluate peft trl bitsandbytes

# 1. Run a phase (example: Phase 1 - Classification)
python scripts/phase/run_phase1.py --config configs/phase1_classification/bert-tweet-sentiment.yaml --exp-id PH1-CLS-001

# 2. Run evaluation for that phase
python scripts/evaluation/run_benchmark.py --phase phase1_classification --model bert-base-uncased --exp-id EVAL-PH1-001

# 3. (Optional) Apply PEFT
python scripts/phase/run_phase1.py --config configs/phase1_classification/bert-tweet-sentiment.yaml --peft lora --exp-id PEFT-PH1-001

# 4. (Optional) Deploy
python scripts/deployment/export_onnx.py --model bert-base-uncased --output-dir tracks/experiments/runs/PH1-CLS-001/onnx/
```

---

## Directory Structure (Minimal & Focused)

```
nlp-with-transformers/
├── phase0_foundations/      # Ch 1: Hello Transformers
├── phase1_classification/   # Ch 2: Text Classification
├── phase2_architecture/     # Ch 3: Transformer Anatomy
├── phase3_ner_multilingual/ # Ch 4: Multilingual NER
├── phase4_generation/       # Ch 5: Text Generation
├── phase5_summarization/    # Ch 6: Summarization
├── phase6_qa_rag/           # Ch 7: Question Answering
├── phase7_efficiency/       # Ch 8: Production Efficiency
│   └── distillation/        # Ch 8.3-8.5: Knowledge distillation deep-dive
├── phase8_low_resource/     # Ch 9: Few/No Labels
├── phase9_from_scratch/     # Ch 10: Training from Scratch
├── phase10_frontiers/       # Ch 11: Future Directions
├── tracks/
│   ├── peft/                # LoRA/QLoRA/AdaLoRA/IA³ per phase
│   ├── evaluation/          # Book-aligned metrics (seqeval, ROUGE, EM/F1, latency)
│   ├── deployment/          # ONNX, GPTQ/AWQ/GGUF, vLLM/TGI
│   └── experiments/         # Unified registry (configs, runs, results, plots)
├── configs/                 # Phase-specific YAML (base + phase + track composition)
├── scripts/                 # Phase entry points + utilities
│   ├── phase/               # run_phase0.py ... run_phase10.py
│   ├── data/                # Download scripts per chapter dataset
│   ├── training/            # Custom trainers (distillation, etc.)
│   ├── evaluation/          # Metric computations
│   ├── deployment/          # Export, quantize, serve
│   ├── analysis/            # Attention viz, layer probes
│   └── utils/               # Config, seeding, device, hub
└── notebooks/               # Phase-aligned exploration notebooks
```

---

## Learning Workflow (Per Phase)

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. READ phase README → understand book chapter & objectives    │
├─────────────────────────────────────────────────────────────────┤
│ 2. RUN book baseline → `python scripts/phase/run_phaseX.py`    │
│    - Uses config from `configs/phaseX_*/`                       │
│    - Logs to `tracks/experiments/runs/PHX-XXX/`                │
├─────────────────────────────────────────────────────────────────┤
│ 3. VERIFY → compare outputs to book (metrics, loss curves)     │
├─────────────────────────────────────────────────────────────────┤
│ 4. EXTEND (optional) → apply tracks:                            │
│    - PEFT: LoRA/QLoRA via `tracks/peft/`                       │
│    - EVAL: Run benchmarks via `tracks/evaluation/`             │
│    - DEPLOY: Export/quantize via `tracks/deployment/`          │
├─────────────────────────────────────────────────────────────────┤
│ 5. LOG → results auto-saved to `tracks/experiments/results/`   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Experiment Tracking (Unified Registry)

All runs (phases + tracks) log to **one registry**:

```
tracks/experiments/
├── configs/          # YAML configs (source of truth)
│   ├── PH1-CLS-001.yaml
│   ├── PEFT-PH1-001.yaml
│   └── EVAL-PH1-001.yaml
├── runs/             # Artifacts (gitignored)
│   └── PH1-CLS-001/{checkpoints,logs,tensorboard,wandb}
├── results/          # Metrics JSON (committed)
│   ├── PH1-CLS-001.json
│   └── PEFT-PH1-001.json
├── plots/            # Visualizations (committed)
│   └── PH1-CLS-001/{loss_curve.png,confusion_matrix.png}
└── registry.csv      # Master index
```

**Experiment ID Format**: `{TRACK}-{PHASE}-{TASK}-{NNN}`
- `PH1-CLS-001` = Phase 1 Classification baseline
- `PEFT-PH1-001` = Phase 1 with LoRA
- `EVAL-PH1-001` = Phase 1 evaluation
- `DEP-PH7-001` = Phase 7 deployment

---

## Book Code Patterns Reference

Each phase README contains **exact code patterns** from the book to reproduce:

| Chapter | Key Patterns |
|---------|--------------|
| **Ch 1** | `pipeline()`, `AutoModel`, `push_to_hub()` |
| **Ch 2** | `TweetEval` load → tokenize → feature extraction → `Trainer` fine-tune |
| **Ch 3** | `nn.MultiheadAttention`, `EncoderBlock`, `sinusoidal_pos_embed()` |
| **Ch 4** | `wikiann` load → `word_ids()` alignment → `AutoModelForTokenClassification` |
| **Ch 5** | `model.generate(do_sample=True, top_k=50, top_p=0.9, temperature=0.7)` |
| **Ch 6** | `Seq2SeqTrainer`, `ROUGE` with `nltk.sent_tokenize` |
| **Ch 7** | Haystack `BM25Retriever` + `FARMReader` + `ExtractiveQAPipeline` |
| **Ch 8** | `DistillationTrainer` (KL loss), `torch.quantization`, `torch.onnx.export` |
| **Ch 9** | `SentenceTransformer` + FAISS k-NN, `SetFit`, few-shot `Trainer` |
| **Ch 10** | `ByteLevelBPETokenizer`, `DataCollatorForLanguageModeling`, scaling law fit |
| **Ch 11** | `flash_attn_func`, `timm.create_model("vit_...")`, `CLIPModel`, `Whisper` |

---

## Modern Extensions (Beyond Book)

| Area | Book (2022) | This Repo Adds |
|------|-------------|----------------|
| **PEFT** | — | LoRA, QLoRA, AdaLoRA, IA³, prefix-tuning |
| **Quantization** | INT8, ONNX | GPTQ, AWQ, GGUF (4-bit) |
| **Serving** | — | vLLM, TGI, llama.cpp, ONNX Runtime |
| **Evaluation** | Basic `evaluate` | `lm-eval-harness`, `RAGAS`, `AlpacaEval`, `MT-Bench` |
| **Tracking** | Manual | `wandb` + custom registry + TensorBoard |
| **Training** | `Trainer` | `accelerate` + FSDP/DeepSpeed, FlashAttention |

---

## Recommended Order

```
Phase 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10
    │      │    │    │    │    │    │    │    │    │    │
    └──────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘
           Run PEFT + Evaluation tracks in parallel
```

**Milestones**:
- **Phase 1-3**: Encoder mastery (BERT, XLM-R)
- **Phase 4-5**: Decoder & Encoder-Decoder (GPT-2, PEGASUS, BART)
- **Phase 6-7**: Production RAG + Efficiency
- **Phase 8-9**: Low-resource + From-scratch
- **Phase 10**: Research frontiers

---

## Commands Cheat Sheet

```bash
# List all phase configs
ls configs/phase1_classification/
ls configs/phase3_ner_multilingual/
# ...

# Run any phase baseline
python scripts/phase/run_phase1.py --config configs/phase1_classification/bert-tweet-sentiment.yaml --exp-id PH1-CLS-001
python scripts/phase/run_phase3.py --config configs/phase3_ner_multilingual/xlm-roberta-wikiann-en.yaml --exp-id PH3-NER-001
# ...

# Apply PEFT to any phase
python scripts/phase/run_phase1.py --config configs/phase1_classification/bert-tweet-sentiment.yaml --peft lora --exp-id PEFT-PH1-001

# Evaluate any phase
python scripts/evaluation/run_benchmark.py --phase phase1_classification --model bert-base-uncased --exp-id EVAL-PH1-001

# Deploy any model
python scripts/deployment/export_onnx.py --model bert-base-uncased --output-dir tracks/experiments/runs/PH1-CLS-001/onnx/
python scripts/deployment/quantize_gptq.py --model mistralai/Mistral-7B-v0.1 --bits 4

# View registry
python scripts/utils/experiments.py list
python scripts/utils/experiments.py compare PH1-CLS-001 PEFT-PH1-001
```

---

## Prerequisites

```bash
# Core (book)
pip install transformers datasets tokenizers accelerate evaluate

# Modern extensions
pip install peft trl bitsandbytes optimum[onnxruntime] auto-gptq autoawq

# Serving (optional)
pip install vllm  # or: docker pull ghcr.io/huggingface/text-generation-inference

# Tracking
pip install wandb tensorboard

# Notebooks
pip install jupyterlab ipywidgets matplotlib seaborn plotly nltk seqeval
```

---

## License

MIT — see [`../../LICENSE`](../../LICENSE). Book code: Apache 2.0.

---

**Start here**: Read `phase0_foundations/README.md` → run `scripts/phase/run_phase0.py --demo pipelines`