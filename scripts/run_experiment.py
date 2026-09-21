#!/usr/bin/env python
"""Utility script to run a single experiment."""

import argparse
import yaml
import subprocess
import sys
from pathlib import Path

def run_experiment(exp_id: str, config_path: str = None):
    """Run a single experiment by ID."""
    
    # Find experiment config
    if config_path is None:
        config_path = Path(f"experiments/configs/{exp_id}.yaml")
    
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    print(f"Running experiment: {exp_id}")
    print(f"Description: {config['experiment']['description']}")
    
    # Run training with experiment config
    cmd = [
        sys.executable, "-m", "hardik_llm.training.train",
        "--experiment-config", str(config_path),
    ]
    
    result = subprocess.run(cmd)
    return result.returncode

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_id", help="Experiment ID (e.g., EXP-001)")
    parser.add_argument("--config", help="Path to experiment config")
    args = parser.parse_args()
    
    sys.exit(run_experiment(args.exp_id, args.config))