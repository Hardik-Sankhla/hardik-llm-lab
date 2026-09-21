#!/usr/bin/env python
"""Utility script to set up data for experiments."""

import os
import sys
import argparse

def download_wikitext(data_dir: str = "data"):
    """Download WikiText-2 dataset."""
    os.makedirs(data_dir, exist_ok=True)
    
    # Using torchtext or datasets library
    try:
        from datasets import load_dataset
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        
        # Save as text files
        for split in ['train', 'validation', 'test']:
            text = "\n".join(dataset[split]['text'])
            with open(os.path.join(data_dir, f"wikitext-2-{split}.txt"), "w") as f:
                f.write(text)
        
        print(f"Downloaded WikiText-2 to {data_dir}/")
    except ImportError:
        print("Install datasets: pip install datasets")
        sys.exit(1)

def prepare_tokens(data_dir: str = "data", vocab_size: int = 50257):
    """Tokenize data using tiktoken (for reference) or custom tokenizer."""
    import tiktoken
    
    enc = tiktoken.get_encoding("gpt2")
    
    for split in ['train', 'validation', 'test']:
        path = os.path.join(data_dir, f"wikitext-2-{split}.txt")
        if not os.path.exists(path):
            print(f"Missing {path}, run download first")
            continue
        
        with open(path) as f:
            text = f.read()
        
        tokens = enc.encode(text)
        torch.save(tokens, os.path.join(data_dir, f"wikitext-2-{split}.pt"))
        print(f"Tokenized {split}: {len(tokens):,} tokens")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--tokenize", action="store_true")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    
    if args.download:
        download_wikitext(args.data_dir)
    if args.tokenize:
        prepare_tokens(args.data_dir)