"""Experiment running and analysis infrastructure.

Implement in guide/09-experiment-infra/
"""

import yaml
import mlflow
from dataclasses import dataclass, asdict
from typing import Optional
import subprocess
import json
from pathlib import Path


@dataclass
class ExperimentRun:
    exp_id: str
    config: dict
    run_id: Optional[str] = None
    status: str = "pending"  # pending, running, completed, failed
    metrics: Optional[dict] = None
    artifacts: Optional[list] = None


def flatten_dict(d: dict, parent_key: str = '', sep: str = '.') -> dict:
    """Flatten nested dict for MLflow logging."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            items.append((new_key, str(v)))
        else:
            items.append((new_key, v))
    return dict(items)


def run_experiment(exp_config_path: str) -> ExperimentRun:
    """Run a single experiment from config."""
    with open(exp_config_path) as f:
        config = yaml.safe_load(f)
    
    exp_id = config['experiment']['id']
    
    # MLflow tracking
    mlflow.set_experiment("hardik-llm-lab")
    with mlflow.start_run(run_name=exp_id) as run:
        run_id = run.info.run_id
        
        # Log config
        mlflow.log_params(flatten_dict(config))
        
        # Run training script
        cmd = [
            "python", "-m", "hardik_llm.training.train",
            "--experiment-config", exp_config_path,
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            mlflow.log_param("status", "failed")
            mlflow.log_text(result.stderr, "error.log")
            return ExperimentRun(exp_id, config, run_id, "failed")
        
        # Parse results (simplified - would parse actual output)
        metrics = parse_results(result.stdout)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        
        mlflow.log_param("status", "completed")
        return ExperimentRun(exp_id, config, run_id, "completed", metrics)


def parse_results(output: str) -> dict:
    """Parse training output for metrics."""
    # Simplified - would extract final val_loss, val_ppl, etc.
    metrics = {}
    for line in output.split('\n'):
        if 'val_loss=' in line:
            # Extract metrics from log lines
            pass
    return metrics


def load_experiment_results(exp_id: str):
    """Load all runs for an experiment from MLflow."""
    runs = mlflow.search_runs(
        experiment_names=["hardik-llm-lab"],
        filter_string=f"params.experiment_id = '{exp_id}'"
    )
    return runs