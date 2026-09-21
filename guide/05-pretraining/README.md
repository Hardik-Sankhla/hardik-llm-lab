# Guide: Pretraining Pipeline (Chapter 5)

**Objective**: Build a complete training pipeline. Implement data loading, optimizer, scheduler, mixed precision, checkpointing, and logging. Train a model and measure everything.

---

## Prerequisites

- Read: `reference/raschka/ch05/01_main-chapter-code/ch05.ipynb`
- Reference: `reference/raschka/ch05/01_main-chapter-code/gpt_train.py`
- Key concepts: DataLoader, AdamW, cosine decay, gradient accumulation, mixed precision, gradient clipping

---

## Your Implementation Checklist

### 1. Training Config (`src/hardik_llm/training/config.py`)

```python
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    # Data
    batch_size: int = 32
    max_seq_len: int = 1024
    gradient_accumulation_steps: int = 1
    num_workers: int = 4
    
    # Optimizer
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    
    # Scheduler
    max_steps: int = 100000
    warmup_steps: int = 1000
    min_lr: float = 3e-5
    
    # Precision & Stability
    mixed_precision: bool = True  # FP16/BF16
    grad_clip: float = 1.0
    
    # Logging & Checkpointing
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 5000
    checkpoint_dir: str = "checkpoints"
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True
```

### 2. Data Loader (`src/hardik_llm/training/data.py`)

```python
import torch
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, tokens: list[int], max_seq_len: int):
        self.tokens = tokens
        self.max_seq_len = max_seq_len
    
    def __len__(self):
        return max(0, len(self.tokens) - self.max_seq_len)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.tokens[idx:idx + self.max_seq_len], dtype=torch.long)
        y = torch.tensor(self.tokens[idx + 1:idx + 1 + self.max_seq_len], dtype=torch.long)
        return x, y

def create_dataloaders(train_tokens, val_tokens, config: TrainingConfig):
    train_ds = TextDataset(train_tokens, config.max_seq_len)
    val_ds = TextDataset(val_tokens, config.max_seq_len)
    
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True, drop_last=False
    )
    return train_loader, val_loader
```

### 3. Optimizer & Scheduler (`src/hardik_llm/training/optimizer.py`)

```python
def create_optimizer(model: nn.Module, config: TrainingConfig):
    # Separate weight decay for different param types
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'bias' in name or 'ln' in name.lower() or 'norm' in name.lower() or 'embedding' in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    optimizer = torch.optim.AdamW(
        [
            {'params': decay_params, 'weight_decay': config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ],
        lr=config.learning_rate, betas=(config.beta1, config.beta2), eps=config.eps
    )
    return optimizer

def create_scheduler(optimizer, config: TrainingConfig):
    def lr_lambda(step):
        if step < config.warmup_steps:
            return step / config.warmup_steps
        # Cosine decay
        progress = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
        progress = min(progress, 1.0)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return config.min_lr / config.learning_rate + (1 - config.min_lr / config.learning_rate) * cosine
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

### 4. Trainer (`src/hardik_llm/training/trainer.py`)

```python
class Trainer:
    def __init__(self, model, train_loader, val_loader, config: TrainingConfig):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        self.optimizer = create_optimizer(model, config)
        self.scheduler = create_scheduler(self.optimizer, config)
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.mixed_precision)
        
        self.step = 0
        self.best_val_loss = float('inf')
        
        # Logging
        self.writer = SummaryWriter(log_dir=f"runs/{config.run_name}")
    
    def train_step(self, batch):
        x, y = batch
        x, y = x.to(device), y.to(device)
        
        with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
            logits, loss = self.model(x, y)
            loss = loss / self.config.gradient_accumulation_steps
        
        self.scaler.scale(loss).backward()
        
        if (self.step + 1) % self.config.gradient_accumulation_steps == 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            self.scheduler.step()
        
        return loss.item() * self.config.gradient_accumulation_steps
    
    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_loss = 0
        for batch in self.val_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            _, loss = self.model(x, y)
            total_loss += loss.item()
        self.model.train()
        return total_loss / len(self.val_loader)
    
    def save_checkpoint(self, path: str, is_best: bool = False):
        torch.save({
            'step': self.step,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }, path)
    
    def train(self):
        self.model.train()
        for epoch in range(100):  # Loop until max_steps
            for batch in self.train_loader:
                loss = self.train_step(batch)
                self.step += 1
                
                if self.step % self.config.log_interval == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    self.writer.add_scalar('train/loss', loss, self.step)
                    self.writer.add_scalar('train/lr', lr, self.step)
                    self.writer.add_scalar('train/grad_norm', get_grad_norm(self.model), self.step)
                
                if self.step % self.config.eval_interval == 0:
                    val_loss = self.evaluate()
                    self.writer.add_scalar('val/loss', val_loss, self.step)
                    self.writer.add_scalar('val/perplexity', math.exp(val_loss), self.step)
                    
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint(f"{self.config.checkpoint_dir}/best.pt", is_best=True)
                
                if self.step % self.config.save_interval == 0:
                    self.save_checkpoint(f"{self.config.checkpoint_dir}/step_{self.step}.pt")
                
                if self.step >= self.config.max_steps:
                    return
```

---

## Experiments to Run (Guide/05-pretraining/)

### EXP-015: Learning Rate Schedule Comparison

**Question**: Cosine decay vs constant vs linear decay vs cosine with restarts?

**Variables**: Different schedulers, same model/data

**Metrics**: Final val loss, convergence speed, stability

---

### EXP-016: Mixed Precision (FP32 vs FP16 vs BF16)

**Question**: Does mixed precision affect final quality?

**Variables**: Precision modes

**Metrics**: Training speed, memory, final val loss, gradient overflow frequency

---

### EXP-017: Gradient Accumulation vs Large Batch

**Question**: Gradient accumulation (effective batch) vs true large batch?

**Variables**:
- batch=8, accum=4 (effective 32)
- batch=32, accum=1

**Metrics**: Training speed, memory, final loss

---

### EXP-018: Weight Decay Values

**Question**: Optimal weight decay for small models?

**Variables**: wd=[0.0, 0.01, 0.1, 0.3]

**Metrics**: Val loss, overfitting gap (train - val loss)

---

## What to Observe & Note

### During Implementation
- [ ] Gradient accumulation: loss scaling correct?
- [ ] Gradient clipping: when does it trigger?
- [ ] Mixed precision: scaler behavior, overflow detection
- [ ] Checkpointing: can resume exactly?

### During Experiments
- [ ] Plot LR schedule vs loss curve
- [ ] Track gradient norm over training (should stabilize)
- [ ] Monitor GPU utilization (should be >90%)
- [ ] Compare FP32 vs FP16 final loss (should be ~same)

### Failure Modes
- [ ] Loss spikes → gradient clipping too low or LR too high
- [ ] NaN → mixed precision overflow, increase loss scale
- [ ] Slow training → data loading bottleneck, increase num_workers
- [ ] Overfitting → increase weight decay, add dropout

---

## Validation Tests (`tests/test_training.py`)

```python
def test_train_step_decreases_loss():
    """Single train step reduces loss on small batch."""
    pass

def test_gradient_accumulation():
    """accum=4 gives same result as batch*4 without accum."""
    pass

def test_mixed_precision_no_nan():
    """FP16 training doesn't produce NaN."""
    pass

def test_checkpoint_resume():
    """Resume from checkpoint continues exactly."""
    pass

def test_scheduler_lr_values():
    """LR follows warmup + cosine exactly."""
    pass

def test_grad_clip_triggers():
    """Gradient clipping actually limits norm."""
    pass
```

---

## Documentation to Write

1. `docs/training/pretraining.md` - Complete pipeline with diagrams
2. `docs/training/optimization.md` - Optimizer, scheduler, precision details
3. `docs/training/stability.md` - Gradient clipping, NaN handling, debugging
4. `guide/05-pretraining/EXPERIMENT_LOG.md`
5. `experiments/configs/EXP-015-lr-schedule.yaml`, etc.
6. Update `docs/research/experiments.md`

---

## Next Step

After pretraining: `guide/06-finetuning/`