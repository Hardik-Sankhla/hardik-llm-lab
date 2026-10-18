# Phase 7: Efficiency & Deployment — Chapter 8

> **Book**: Ch 8 "Making Transformers Efficient in Production" (pp. 233–272)  
> **Core**: Intent detection case study — distillation, quantization, ONNX, pruning, benchmarking

## Book Implementation Checklist

| Section | Book Task | Key Code | Extension Ideas |
|---------|-----------|----------|-----------------|
| 8.1 | Intent detection dataset | `load_dataset("clinc_oos", "small")` | Other intent datasets (SNIPS, BANKING77) |
| 8.2 | Performance benchmark | `transformers.utils.benchmark` | Latency, throughput, memory |
| 8.3 | Knowledge distillation | `Trainer` with custom loss | Response/feature/attention distillation |
| 8.4 | Distillation for FT | DistilBERT student from BERT teacher | Optuna hyperparameter search |
| 8.5 | Distillation for pretraining | DistilGPT-2 from GPT-2 | Large-scale distillation |
| 8.6 | Quantization | `torch.quantization`, ONNX Runtime | Dynamic vs static vs GPTQ/AWQ |
| 8.7 | ONNX export | `torch.onnx.export` | ONNX Runtime optimization |
| 8.8 | Weight pruning | `torch.nn.utils.prune` | Structured vs unstructured |

## Recommended Experiments

| Exp ID | Variant | Teacher | Student | Method | Target Speedup |
|--------|---------|---------|---------|--------|----------------|
| PH7-001 | Book baseline | — | `distilbert-base-uncased` | Full FT | 1.5x |
| PH7-002 | Distillation (FT) | `bert-base-uncased` | `distilbert-base-uncased` | Response-based | 1.8x |
| PH7-003 | Distillation + Optuna | `bert-base-uncased` | `distilbert-base-uncased` | Response + HPO | 2.0x |
| PH7-004 | Dynamic quantization | — | `bert-base-uncased` | INT8 dynamic | 1.5x |
| PH7-005 | Static quantization | — | `bert-base-uncased` | INT8 static | 2.0x |
| PH7-006 | ONNX Runtime | — | `bert-base-uncased` | ONNX + ORT | 2.5x |
| PH7-007 | GPTQ (modern) | — | `mistral-7b` | 4-bit GPTQ | 3.5x |
| PH7-008 | Pruning | — | `bert-base-uncased` | 50% sparsity | 1.3x |

## Quick Start

```bash
# Book: Intent detection baseline (Ch 8.1-8.2)
python -m phase7_efficiency.run \
  --config configs/phase7/intent-baseline.yaml \
  --exp-id PH7-001

# Book: Knowledge distillation (Ch 8.3-8.5)
python -m phase7_efficiency.run \
  --config configs/phase7/distill-intent.yaml \
  --method distillation \
  --teacher bert-base-uncased \
  --student distilbert-base-uncased \
  --exp-id PH7-002

# Book: Distillation with Optuna HPO (Ch 8.5)
python -m phase7_efficiency.run \
  --config configs/phase7/distill-intent-optuna.yaml \
  --method distillation \
  --hpo \
  --exp-id PH7-003

# Book: Quantization (Ch 8.6-8.7)
python -m phase7_efficiency.run \
  --config configs/phase7/quantize-intent.yaml \
  --method quantization \
  --quant-type dynamic_int8 \
  --exp-id PH7-004

python -m phase7_efficiency.run \
  --config configs/phase7/quantize-intent.yaml \
  --method quantization \
  --quant-type static_int8 \
  --exp-id PH7-005

# Book: ONNX export (Ch 8.7)
python -m phase7_efficiency.run \
  --config configs/phase7/onnx-intent.yaml \
  --method onnx \
  --exp-id PH7-006

# Modern: GPTQ quantization (beyond book)
python -m phase7_efficiency.run \
  --config configs/phase7/gptq-mistral.yaml \
  --method gptq \
  --bits 4 \
  --exp-id PH7-007
```

## Key Book Code Patterns to Reproduce

```python
# 1. Performance benchmark (Ch 8.2)
from transformers.utils import benchmark
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
benchmark_results = benchmark(
    model, batch_size=1, sequence_length=128,
    num_iterations=100, warmup=10
)
print(f"Latency: {benchmark_results.latency.mean:.2f}ms")
print(f"Throughput: {benchmark_results.throughput.mean:.2f} samples/s")

# 2. Knowledge distillation trainer (Ch 8.3-8.4)
from transformers import Trainer, TrainingArguments
import torch.nn.functional as F

class DistillationTrainer(Trainer):
    def __init__(self, teacher_model, temperature=2.0, alpha=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.temperature = temperature
        self.alpha = alpha

    def compute_loss(self, model, inputs, return_outputs=False):
        student_outputs = model(**inputs)
        with torch.no_grad():
            teacher_outputs = self.teacher(**inputs)

        # Hard loss (CE with labels)
        loss_ce = student_outputs.loss

        # Soft loss (KL divergence between student/teacher logits)
        loss_kl = F.kl_div(
            F.log_softmax(student_outputs.logits / self.temperature, dim=-1),
            F.softmax(teacher_outputs.logits / self.temperature, dim=-1),
            reduction="batchmean"
        ) * (self.temperature ** 2)

        loss = self.alpha * loss_ce + (1 - self.alpha) * loss_kl
        return (loss, student_outputs) if return_outputs else loss

# 3. Quantization (Ch 8.6)
# Dynamic quantization (easiest)
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Static quantization (requires calibration)
model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
torch.quantization.prepare(model, inplace=True)
# ... calibrate with representative data ...
torch.quantization.convert(model, inplace=True)

# 4. ONNX export (Ch 8.7)
import torch.onnx
dummy_input = tokenizer("Hello world", return_tensors="pt")
torch.onnx.export(
    model, (dummy_input["input_ids"], dummy_input["attention_mask"]),
    "model.onnx", input_names=["input_ids", "attention_mask"],
    output_names=["logits"], dynamic_axes={
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        "logits": {0: "batch", 1: "seq"}
    }, opset_version=14
)

# 5. Weight pruning (Ch 8.8)
import torch.nn.utils.prune as prune
for module in model.modules():
    if isinstance(module, torch.nn.Linear):
        prune.l1_unstructured(module, name="weight", amount=0.5)
# Make pruning permanent
for module in model.modules():
    if isinstance(module, torch.nn.Linear):
        prune.remove(module, "weight")
```

## Efficiency Techniques Comparison (Book + Modern)

| Technique | Speedup | Quality Loss | Complexity | Book Ch | Modern Tools |
|-----------|---------|--------------|------------|---------|--------------|
| **Distillation** | 1.5-2x | 1-3% | Medium | 8.3-8.5 | Custom trainer |
| **Quantization (INT8)** | 1.5-2x | 1-2% | Low | 8.6 | PyTorch, ONNX RT |
| **Quantization (GPTQ/AWQ)** | 3-4x | 1-2% | Medium | — | `auto-gptq`, `autoawq` |
| **ONNX Runtime** | 2-3x | 0-1% | Medium | 8.7 | `optimum.onnx` |
| **Pruning** | 1.2-1.5x | 2-5% | High | 8.8 | `torch.prune`, `sparseml` |
| **Speculative Decoding** | 2-3x | 0% | High | — | `vLLM`, `transformers` |

## Evaluation Protocol

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Latency (p50)** | <50ms | Batch=1, seq=128 |
| **Throughput** | >100 samples/s | Batch=32, seq=128 |
| **Memory** | <2GB GPU | Peak VRAM |
| **Accuracy Drop** | <2% F1 | vs full-precision baseline |
| **Model Size** | <500MB | Disk footprint |

## Modern Extensions (Beyond Book)

| Technique | Library | Use Case |
|-----------|---------|----------|
| **GPTQ** | `auto-gptq` | 4-bit LLM quantization |
| **AWQ** | `autoawq` | Activation-aware quantization |
| **GGUF** | `llama.cpp` | CPU/edge deployment |
| **TensorRT-LLM** | NVIDIA | Max GPU throughput |
| **vLLM** | `vllm` | High-throughput serving |
| **SparseML** | `neuralmagic` | Structured pruning + quantization |

## Next Phase Dependencies

- Phase 9 (From Scratch): Apply efficiency to custom-trained models
- Tracks/Deployment: Export optimized models for serving
- Tracks/PEFT: Combine LoRA + quantization (QLoRA)