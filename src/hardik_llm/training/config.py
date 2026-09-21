"""Training configuration and components.

Implement in guide/05-pretraining/
"""

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
    mixed_precision: bool = True
    grad_clip: float = 1.0
    
    # Logging & Checkpointing
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 5000
    checkpoint_dir: str = "checkpoints"
    run_name: str = "default"
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True