# PEFT Track — Parameter-Efficient Fine-Tuning

> **Book Context**: Ch 9 (Few-shot learning), Ch 8 (Efficiency) — modern PEFT extends both  
> **Core**: LoRA, QLoRA, AdaLoRA, IA³ — run across Phases 1, 3, 4, 5, 6, 7, 8

## Book Connection

| Book Chapter | Book Technique | Modern PEFT Equivalent |
|--------------|----------------|------------------------|
| Ch 9.4 | Few-shot fine-tuning | LoRA (fewer params, faster) |
| Ch 9.7 | In-context learning | Prompt/Prefix tuning (soft prompts) |
| Ch 8.3 | Knowledge distillation | LoRA + distillation (student init) |
| Ch 8.6 | Quantization | QLoRA (4-bit + LoRA) |

## Methods (Modern Extensions)

| Method | Trainable Params | Storage | Speed | Quality | Best Phase |
|--------|------------------|---------|-------|---------|------------|
| **LoRA** | ~0.1-1% | Low | Fast | ★★★★★ | All |
| **QLoRA** | ~0.1% (4-bit) | Very Low | Fast | ★★★★☆ | 4, 6, 7, 9 |
| **AdaLoRA** | Adaptive budget | Low | Fast | ★★★★☆ | 1, 3, 5 |
| **IA³** | ~0.01% | Very Low | Fast | ★★★★☆ | 8, 9 |
| **Prefix Tuning** | ~0.1% | Low | Fast | ★★★☆☆ | 4, 6 |
| **Prompt Tuning** | ~0.01% | Very Low | Fast | ★★★☆☆ | 8 |
| **P-Tuning v2** | ~0.1% | Low | Fast | ★★★★☆ | 4, 6 |

## LoRA Config (Per Phase)

```yaml
# Phase 1 (Classification) - BERT
peft:
  method: lora
  r: 16
  alpha: 32
  target_modules: ["query", "value", "key", "dense"]
  task_type: "SEQ_CLS"

# Phase 3 (NER) - XLM-RoBERTa
peft:
  method: lora
  r: 16
  alpha: 32
  target_modules: ["query", "value", "key", "dense"]
  task_type: "TOKEN_CLS"

# Phase 4 (Generation) - GPT-2
peft:
  method: lora
  r: 16
  alpha: 32
  target_modules: ["c_attn", "c_proj", "c_fc", "c_proj"]
  task_type: "CAUSAL_LM"

# Phase 5 (Summarization) - PEGASUS/BART
peft:
  method: lora
  r: 32
  alpha: 64
  target_modules: ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"]
  task_type: "SEQ_2_SEQ_LM"

# Phase 6 (QA) - DeBERTa/RoBERTa
peft:
  method: lora
  r: 16
  alpha: 32
  target_modules: ["query_proj", "value_proj", "key_proj", "output_dense"]
  task_type: "QUESTION_ANS"

# Phase 7 (Efficiency) - DistilBERT
peft:
  method: lora
  r: 8
  alpha: 16
  target_modules: ["q_lin", "v_lin", "k_lin", "out_lin"]
  task_type: "SEQ_CLS"
```

## QLoRA Config (4-bit + LoRA)

```yaml
peft:
  method: lora
  r: 64
  alpha: 128
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
  quantization:
    load_in_4bit: true
    bnb_4bit_compute_dtype: "bfloat16"
    bnb_4bit_quant_type: "nf4"
    bnb_4bit_use_double_quant: true
```

## Quick Start

```bash
# Phase 1: Classification with LoRA
python -m tracks.peft.run \
  --phase phase1_classification \
  --config configs/peft/lora-classification.yaml \
  --exp-id PEFT-PH1-001

# Phase 3: NER with LoRA
python -m tracks.peft.run \
  --phase phase3_ner_multilingual \
  --config configs/peft/lora-ner.yaml \
  --exp-id PEFT-PH3-001

# Phase 4: Generation with QLoRA (Mistral-7B)
python -m tracks.peft.run \
  --phase phase4_generation \
  --config configs/peft/qlora-generation.yaml \
  --model mistralai/Mistral-7B-v0.1 \
  --exp-id PEFT-PH4-001

# Phase 5: Summarization with LoRA (PEGASUS)
python -m tracks.peft.run \
  --phase phase5_summarization \
  --config configs/peft/lora-summarization.yaml \
  --exp-id PEFT-PH5-001

# Phase 6: QA with LoRA
python -m tracks.peft.run \
  --phase phase6_qa_rag \
  --config configs/peft/lora-qa.yaml \
  --exp-id PEFT-PH6-001

# Phase 8: Few-shot with IA³
python -m tracks.peft.run \
  --phase phase8_low_resource \
  --config configs/peft/ia3-fewshot.yaml \
  --exp-id PEFT-PH8-001

# Compare all methods on same task
python -m tracks.peft.compare \
  --phase phase1_classification \
  --methods lora,qlora,adalora,ia3,prefix \
  --exp-id PEFT-COMPARE-001
```

## Cross-Phase PEFT Experiments

| Experiment | Phases | Goal |
|------------|--------|------|
| **LoRA rank sweep** | 1, 3, 4, 5, 6 | Find optimal r per task |
| **QLoRA vs LoRA** | 4, 6, 9 | Quality vs memory tradeoff |
| **IA³ for few-shot** | 8 | Minimal params for low-resource |
| **LoRA + Distillation** | 7 | Student init from LoRA teacher |
| **Continual LoRA** | All | Add tasks without forgetting |

## Evaluation Benchmarks

| Benchmark | Phases | Metric |
|-----------|--------|--------|
| GLUE | 1, 7, 8 | Avg score |
| SuperGLUE | 1, 8 | Avg score |
| WikiANN NER | 3 | F1 |
| SQuAD 2.0 | 6 | EM, F1 |
| CNN/DailyMail | 5 | ROUGE-L |
| SAMSum | 5 | ROUGE-L |
| AlpacaEval | 4 | Win rate |
| MT-Bench | 4 | GPT-4 score |
| MMLU | 4, 9 | Accuracy |

## Key Code Pattern

```python
from peft import LoraConfig, get_peft_model, TaskType

def apply_lora(model, phase_config):
    config = LoraConfig(
        r=phase_config.r,
        lora_alpha=phase_config.alpha,
        lora_dropout=phase_config.dropout,
        target_modules=phase_config.target_modules,
        bias=phase_config.bias,
        task_type=TaskType[phase_config.task_type],
    )
    return get_peft_model(model, config)

# For QLoRA
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name, quantization_config=bnb_config, device_map="auto"
)
model = apply_lora(model, phase_config)
```

## Integration with Tracks

| Track | Integration |
|-------|-------------|
| **Evaluation** | Run PEFT variants on all benchmarks |
| **Deployment** | Export LoRA adapters separately, merge for serving |
| **Experiments** | Track PEFT configs in registry (PEFT-PH{X}-{NNN}) |