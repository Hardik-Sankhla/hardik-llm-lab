"""CLI entry points for hardik_llm."""

import argparse
import sys

def main():
    parser = argparse.ArgumentParser(prog="hardik-llm")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--config", required=True, help="Training config YAML")
    train_parser.add_argument("--model-config", required=True, help="Model config YAML")
    train_parser.add_argument("--experiment-config", help="Experiment config YAML")
    train_parser.add_argument("--resume", help="Resume from checkpoint")
    
    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate text")
    gen_parser.add_argument("--checkpoint", required=True, help="Model checkpoint")
    gen_parser.add_argument("--prompt", required=True, help="Generation prompt")
    gen_parser.add_argument("--max-new-tokens", type=int, default=100)
    gen_parser.add_argument("--temperature", type=float, default=0.8)
    gen_parser.add_argument("--top-k", type=int, default=50)
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model")
    eval_parser.add_argument("--checkpoint", required=True)
    eval_parser.add_argument("--data", required=True)
    eval_parser.add_argument("--benchmarks", action="store_true")
    
    # Experiment command
    exp_parser = subparsers.add_parser("experiment", help="Run experiment")
    exp_parser.add_argument("--exp-id", required=True)
    exp_parser.add_argument("--config", help="Experiment config")
    
    args = parser.parse_args()
    
    if args.command == "train":
        from hardik_llm.training.train import main as train_main
        train_main(args)
    elif args.command == "generate":
        from hardik_llm.generation.generate import main as gen_main
        gen_main(args)
    elif args.command == "evaluate":
        from hardik_llm.evaluation.evaluate import main as eval_main
        eval_main(args)
    elif args.command == "experiment":
        from hardik_llm.experiments.run import main as exp_main
        exp_main(args)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()