"""Generation CLI."""

import argparse
import torch
from hardik_llm.model import GPTModel, GPTConfig
from hardik_llm.tokenizer import BPETokenizer, TokenizerConfig


def main(args):
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    config = checkpoint['config'] if 'config' in checkpoint else GPTConfig()
    model = GPTModel(config)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load tokenizer
    tokenizer = BPETokenizer(TokenizerConfig())
    # TODO: Load trained tokenizer from checkpoint
    
    # Generate
    device = next(model.parameters()).device
    input_ids = torch.tensor([tokenizer.encode(args.prompt)], device=device)
    generated = model.generate(input_ids, args.max_new_tokens, args.temperature, args.top_k)
    text = tokenizer.decode(generated[0].tolist())
    print(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    main(parser.parse_args())