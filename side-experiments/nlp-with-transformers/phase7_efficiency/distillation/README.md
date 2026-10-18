# Knowledge Distillation — Chapter 8.3–8.5

> **Book**: Ch 8 "Making Transformers Efficient in Production" — Distillation sections (pp. 241–253)  
> **Core**: Response-based distillation for fine-tuning, Optuna HPO, benchmarking

## Book Implementation Checklist

| Section | Book Task | Key Code | Extension Ideas |
|---------|-----------|----------|-----------------|
| 8.3 | Distillation for fine-tuning | Custom `Trainer` with KL loss | Feature-based, attention-based |
| 8.4 | Choosing student init | Pre-trained vs random init | Layer mapping strategies |
| 8.5 | Optuna hyperparameter search | `optuna.create_study()` | T, α, learning rate, batch size |
| 8.5 | Benchmarking distilled model | Latency, throughput, accuracy | Compare to quantized/pruned |

## Distillation Methods (Book + Extensions)

| Method | Loss Components | Book Coverage | Best For |
|--------|-----------------|---------------|----------|
| **Response-based** | CE + α·KL(logits_T, logits_S) | Ch 8.3 | Classification, QA |
| **Feature-based** | CE + β·MSE(hidden_T, hidden_S) | — | All tasks |
| **Attention-based** | CE + γ·MSE(attn_T, attn_S) | — | Structured prediction |
| **Progressive** | Layer-wise distillation | — | Deep students |
| **Data-free** | Generate from teacher | — | No data access |

## Teacher → Student Pairs (Book + Modern)

| Teacher | Student | Init Strategy | Task | Book Exp |
|---------|---------|---------------|------|----------|
| `bert-base-uncased` | `distilbert-base-uncased` | Pre-trained | Intent detection | Ch 8.4 |
| `bert-large-uncased` | `bert-base-uncased` | Pre-trained | GLUE | — |
| `roberta-large` | `distilroberta-base` | Pre-trained | NER, QA | — |
| `bart-large-cnn` | `distilbart-cnn-12-6` | Pre-trained | Summarization | — |
| `flan-t5-xl` | `flan-t5-base` | Pre-trained | Generative QA | — |
| `llama-2-13b` | `llama-2-7b` | Pre-trained | Generation | — |

## Quick Start

```bash
# Book: Response-based distillation (Ch 8.3)
python -m phase7_efficiency.distillation.run \
  --config configs/phase7/distillation/response-based.yaml \
  --teacher bert-base-uncased \
  --student distilbert-base-uncased \
  --dataset clinc_oos \
  --exp-id PH7-DST-001

# Book: Feature-based distillation
python -m phase7_efficiency.distillation.run \
  --config configs/phase7/distillation/feature-based.yaml \
  --teacher bert-base-uncased \
  --student distilbert-base-uncased \
  --exp-id PH7-DST-002

# Book: Optuna HPO (Ch 8.5)
python -m phase7_efficiency.distillation.run \
  --config configs/phase7/distillation/optuna.yaml \
  --hpo \
  --n-trials 50 \
  --exp-id PH7-DST-003

# Modern: LLM distillation (beyond book)
python -m phase7_efficiency.distillation.run \
  --config configs/phase7/distillation/llm-distill.yaml \
  --teacher mistral-7b-instruct \
  --student mistral-7b \
  --method response \
  --exp-id PH7-DST-004
```

## Key Book Code Patterns to Reproduce

```python
# 1. Distillation Trainer (Ch 8.3)
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

        # Hard labels loss
        loss_ce = student_outputs.loss

        # Soft labels loss (KL divergence)
        loss_kl = F.kl_div(
            F.log_softmax(student_outputs.logits / self.temperature, dim=-1),
            F.softmax(teacher_outputs.logits / self.temperature, dim=-1),
            reduction="batchmean"
        ) * (self.temperature ** 2)

        loss = self.alpha * loss_ce + (1 - self.alpha) * loss_kl
        return (loss, student_outputs) if return_outputs else loss

# 2. Feature-based distillation (hidden states)
def compute_feature_loss(student_hidden, teacher_hidden):
    # Align layers if different depths
    losses = []
    for s_h, t_h in zip(student_hidden, teacher_hidden):
        losses.append(F.mse_loss(s_h, t_h))
    return torch.stack(losses).mean()

# 3. Optuna HPO (Ch 8.5)
import optuna

def objective(trial):
    temperature = trial.suggest_float("temperature", 1.0, 5.0)
    alpha = trial.suggest_float("alpha", 0.1, 0.9)
    lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)

    trainer = DistillationTrainer(
        teacher_model=teacher,
        temperature=temperature,
        alpha=alpha,
        args=TrainingArguments(learning_rate=lr, ...)
    )
    trainer.train()
    return trainer.evaluate()["eval_f1"]

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
```

## Optuna Search Space (Book Ch 8.5)

| Parameter | Range | Book Default |
|-----------|-------|--------------|
| `temperature` | [1.0, 5.0] | 2.0 |
| `alpha` (CE weight) | [0.1, 0.9] | 0.5 |
| `learning_rate` | [1e-5, 5e-4] | 2e-5 |
| `batch_size` | [16, 64] | 32 |
| `num_epochs` | [3, 10] | 5 |

## Evaluation Protocol

| Metric | Target | Baseline |
|--------|--------|----------|
| **Task F1** | ≥ Teacher - 2% | Teacher F1 |
| **Latency** | ≥ 1.5x faster | Teacher latency |
| **Model Size** | ≤ 50% teacher | Teacher params |
| **Memory** | ≤ 50% VRAM | Teacher peak |

## Modern Extensions (Beyond Book)

| Technique | Description | Library |
|-----------|-------------|---------|
| **Logit distillation + LoRA** | Distill with LoRA adapters | `peft` + custom |
| **Quantization-aware distillation** | Distill + quantize jointly | Custom |
| **Sequence-level distillation** | For generative tasks | `transformers` |
| **MiniLM-style** | Deep self-attention distillation | Custom |
| **DistilBERT v2** | Improved student initialization | — |