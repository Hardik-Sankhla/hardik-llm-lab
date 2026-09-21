"""Training loop with logging and checkpointing.

Implement in guide/05-pretraining/
"""

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
import math
from pathlib import Path

from hardik_llm.training.config import TrainingConfig
from hardik_llm.training.optimizer import create_optimizer, create_scheduler, get_grad_norm


class Trainer:
    """Complete training loop with mixed precision, logging, checkpointing."""
    
    def __init__(self, model: nn.Module, train_loader, val_loader, config: TrainingConfig, device: str = "cuda"):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        self.model.to(device)
        
        self.optimizer = create_optimizer(model, config)
        self.scheduler = create_scheduler(self.optimizer, config)
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.mixed_precision)
        
        self.step = 0
        self.best_val_loss = float('inf')
        
        # Setup checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Logging
        self.writer = SummaryWriter(log_dir=f"runs/{config.run_name}")
    
    def train_step(self, batch):
        """Single training step with gradient accumulation."""
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        
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
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        for batch in self.val_loader:
            x, y = batch
            x, y = x.to(self.device), y.to(self.device)
            _, loss = self.model(x, y)
            total_loss += loss.item() * x.numel()
            total_tokens += x.numel()
        
        self.model.train()
        return total_loss / total_tokens
    
    def save_checkpoint(self, path: str, is_best: bool = False):
        """Save training checkpoint."""
        checkpoint = {
            'step': self.step,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }
        torch.save(checkpoint, path)
        
        if is_best:
            # Also save as best
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.scaler.load_state_dict(checkpoint['scaler'])
        self.step = checkpoint['step']
        self.best_val_loss = checkpoint['best_val_loss']
        print(f"Resumed from step {self.step}")
    
    def train(self):
        """Main training loop."""
        self.model.train()
        
        while self.step < self.config.max_steps:
            for batch in self.train_loader:
                loss = self.train_step(batch)
                self.step += 1
                
                # Logging
                if self.step % self.config.log_interval == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    grad_norm = get_grad_norm(self.model)
                    self.writer.add_scalar('train/loss', loss, self.step)
                    self.writer.add_scalar('train/lr', lr, self.step)
                    self.writer.add_scalar('train/grad_norm', grad_norm, self.step)
                    print(f"Step {self.step}: loss={loss:.4f}, lr={lr:.2e}, grad_norm={grad_norm:.2f}")
                
                # Evaluation
                if self.step % self.config.eval_interval == 0:
                    val_loss = self.evaluate()
                    val_ppl = math.exp(val_loss)
                    self.writer.add_scalar('val/loss', val_loss, self.step)
                    self.writer.add_scalar('val/perplexity', val_ppl, self.step)
                    print(f"Step {self.step}: val_loss={val_loss:.4f}, val_ppl={val_ppl:.2f}")
                    
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint(
                            self.checkpoint_dir / f"step_{self.step}.pt", 
                            is_best=True
                        )
                
                # Regular checkpoint
                if self.step % self.config.save_interval == 0:
                    self.save_checkpoint(self.checkpoint_dir / f"step_{self.step}.pt")
                
                if self.step >= self.config.max_steps:
                    print(f"Training complete at step {self.step}")
                    return
        
        self.writer.close()