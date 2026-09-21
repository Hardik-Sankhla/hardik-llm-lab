# Guide: Fine-tuning (Chapter 6-7)

**Objective**: Implement classification fine-tuning and instruction fine-tuning. Understand the differences from pretraining. Run controlled experiments on freezing strategies.

---

## Prerequisites

- Read: `reference/raschka/ch06/01_main-chapter-code/ch06.ipynb`
- Read: `reference/raschka/ch07/01_main-chapter-code/ch07.ipynb`
- Reference: `gpt_class_finetune.py`, `gpt_instruction_finetuning.py`
- Key concepts: Head replacement, layer freezing, LoRA, instruction formatting

---

## Your Implementation Checklist

### 1. Classification Fine-tuning (`src/hardik_llm/finetuning/classification.py`)

```python
class ClassificationHead(nn.Module):
    def __init__(self, d_model: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D] -> use last token (CLS-style) or mean pool
        cls_token = x[:, -1, :]  # Last token (GPT-style)
        return self.classifier(self.dropout(cls_token))

def create_classification_model(base_model: GPTModel, num_classes: int, 
                                freeze_backbone: bool = True) -> nn.Module:
    """Replace LM head with classification head."""
    model = copy.deepcopy(base_model)
    model.lm_head = ClassificationHead(model.config.d_model, num_classes)
    
    if freeze_backbone:
        for param in model.embeddings.parameters():
            param.requires_grad = False
        for block in model.blocks:
            for param in block.parameters():
                param.requires_grad = False
        for param in model.ln_f.parameters():
            param.requires_grad = False
    
    return model
```

### 2. Instruction Fine-tuning (`src/hardik_llm/finetuning/instruction.py`)

```python
def format_instruction(example: dict) -> str:
    """Format instruction dataset example."""
    if 'input' in example and example['input']:
        return f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
    return f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"

class InstructionDataset(Dataset):
    def __init__(self, data: list[dict], tokenizer, max_seq_len: int):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = format_instruction(self.data[idx])
        tokens = self.tokenizer.encode(text)
        # Truncate
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        # Pad
        tokens = tokens + [self.tokenizer.pad_token_id] * (self.max_seq_len - len(tokens))
        
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        # Mask padding in loss
        y[y == self.tokenizer.pad_token_id] = -1
        return x, y
```

### 3. Freezing Strategies (`src/hardik_llm/finetuning/strategies.py`)

```python
def freeze_strategy(model: GPTModel, strategy: str):
    """
    Strategies:
    - 'full': Train all params
    - 'head_only': Only classification head
    - 'last_n_layers': Last N transformer blocks + head
    - 'lora': Only LoRA adapters (see guide/07-lora)
    """
    # First freeze all
    for param in model.parameters():
        param.requires_grad = False
    
    if strategy == 'full':
        for param in model.parameters():
            param.requires_grad = True
    
    elif strategy == 'head_only':
        for param in model.lm_head.parameters():
            param.requires_grad = True
    
    elif strategy.startswith('last_'):
        n = int(strategy.split('_')[1])
        # Unfreeze last N blocks + ln_f + head
        for block in model.blocks[-n:]:
            for param in block.parameters():
                param.requires_grad = True
        for param in model.ln_f.parameters():
            param.requires_grad = True
        for param in model.lm_head.parameters():
            param.requires_grad = True
    
    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Strategy '{strategy}': {trainable:,}/{total:,} params trainable ({100*trainable/total:.1f}%)")
```

---

## Experiments to Run (Guide/06-finetuning/)

### EXP-019: Freezing Strategy Comparison

**Question**: How much of the pretrained model needs to be fine-tuned?

**Variables**:
- Strategies: 'full', 'head_only', 'last_1', 'last_2', 'last_4', 'last_6'
- Dataset: Small classification task (e.g., IMDB sentiment, 25k train)

**Metrics**:
- Test accuracy
- Training time
- Trainable parameters
- Catastrophic forgetting (pretrain perplexity after FT)

---

### EXP-020: Instruction Tuning Data Quality

**Question**: How does dataset size/quality affect instruction following?

**Variables**:
- Dataset sizes: [100, 500, 1000, 5000, 20000] examples
- Same base model, same training steps

**Metrics**:
- Qualitative generation quality
- Eval on held-out instructions
- Perplexity on instruction format

---

### EXP-021: Classification vs Instruction Fine-tuning

**Question**: Different objectives, different optimal strategies?

**Compare**: Classification head vs LM head fine-tuning on same base model

---

## What to Observe & Note

### During Implementation
- [ ] Classification: use last token or mean pooling?
- [ ] Instruction formatting: does template matter?
- [ ] Loss masking: padding tokens should be ignored
- [ ] LR for fine-tuning: typically 10x smaller than pretrain

### During Experiments
- [ ] Plot strategy vs accuracy (expect diminishing returns after last_2)
- [ ] Track forgetting: measure pretrain perplexity before/after FT
- [ ] Instruction tuning: monitor generation quality, not just loss
- [ ] Data scaling: log-log plot of data size vs quality

### Failure Modes
- [ ] Catastrophic forgetting: full fine-tuning destroys pretrain knowledge
- [ ] Overfitting: small dataset + full fine-tuning = memorization
- [ ] Generation degradation: instruction tuning breaks coherent generation

---

## Validation Tests (`tests/test_finetuning.py`)

```python
def test_classification_head_shape():
    """ClassificationHead: [B, T, D] -> [B, num_classes]"""
    pass

def test_freeze_strategies():
    """Each strategy produces correct trainable param count."""
    pass

def test_instruction_formatting():
    """Format produces expected string template."""
    pass

def test_loss_masking():
    """Padding tokens don't contribute to loss."""
    pass
```

---

## Documentation to Write

1. `docs/adaptation/classification.md` - Classification FT details
2. `docs/adaptation/instruction-tuning.md` - Instruction FT details
3. `guide/06-finetuning/EXPERIMENT_LOG.md`
4. Update `docs/research/experiments.md`

---

## Next Step

After fine-tuning: `guide/07-lora/`