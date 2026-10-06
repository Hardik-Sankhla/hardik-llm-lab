"""Training CLI entry point."""

import argparse
import yaml
from hardik_llm.training import Trainer, TrainingConfig
from hardik_llm.model import GPTModel, GPTConfig
from hardik_llm.tokenizer import BPETokenizer, TokenizerConfig
from hardik_llm.training.data import create_dataloaders, TextDataset


def main(args):
    # Load configs
    with open(args.model_config) as f:
        model_config = yaml.safe_load(f)
    with open(args.config) as f:
        training_config = yaml.safe_load(f)
    
    model_cfg = GPTConfig(**model_config)
    train_cfg = TrainingConfig(**training_config)
    
    # Model
    model = GPTModel(model_cfg)
    
    # Data - need to train tokenizer first or load
    tokenizer = BPETokenizer(TokenizerConfig())
    # TODO: Train or load tokenizer
    train_tokens = []  # placeholder
    val_tokens = []    # placeholder
    train_loader, val_loader = create_dataloaders(train_tokens, val_tokens, train_cfg)
    
    # Trainer
    trainer = Trainer(model, train_loader, val_loader, train_cfg)
    
    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Training config YAML")
    parser.add_argument("--model-config", required=True, help="Model config YAML")
    parser.add_argument("--experiment-config", help="Experiment config YAML")
    parser.add_argument("--resume", help="Resume from checkpoint")
    main(parser.parse_args())