# Guide: Experiment Infrastructure

**Objective**: Build reproducible experiment tracking, configuration management, and result analysis infrastructure. This is the "systems" layer that makes everything else rigorous.

---

## Prerequisites

- Understanding of: MLflow, W&B, config management, experiment design
- Tools: PyYAML, MLflow, TensorBoard, pandas, matplotlib

---

## Your Implementation Checklist

### 1. Experiment Config Schema (`configs/experiment_schema.yaml`)

```yaml
# Base experiment configuration
experiment:
  id: "EXP-001"
  name: "tokenizer-vocab-size"
  description: "How does vocab size affect compression and downstream loss?"
  author: "Hardik Sankhla"
  created: "2025-01-15"
  
  # Hypothesis
  hypothesis: "Larger vocab reduces sequence length but increases embedding params; optimal around 8k-16k for small models."
  
  # Variables
  variables:
    independent:
      - name: "vocab_size"
        type: "categorical"
        values: [1000, 2000, 4000, 8000, 16000, 32000, 50257]
    dependent:
      - name: "compression_ratio"
        type: "continuous"
      - name: "val_perplexity"
        type: "continuous"
      - name: "training_time_sec"
        type: "continuous"
  
  # Fixed parameters
  fixed_params:
    model_config: "configs/model_small.yaml"
    training_config: "configs/train_short.yaml"
    dataset: "data/wikitext-10mb.txt"
    seed: 42
  
  # Compute
  compute:
    device: "cuda"
    gpu_type: "T4"
    estimated_time_per_run: "30min"
```

### 2. Experiment Runner (`src/hardik_llm/experiments/run.py`)

```python
import mlflow
import yaml
from dataclasses import dataclass, asdict
import subprocess
import json

@dataclass
class ExperimentRun:
    exp_id: str
    config: dict
    run_id: str = None
    status: str = "pending"  # pending, running, completed, failed
    metrics: dict = None
    artifacts: list = None

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
            "--config", config['experiment']['fixed_params']['training_config'],
            "--model-config", config['experiment']['fixed_params']['model_config'],
            "--exp-id", exp_id,
        ]
        
        # Inject variable values
        for var in config['experiment']['variables']['independent']:
            cmd.extend([f"--{var['name']}", str(var['values'][0])])  # Simplified
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            mlflow.log_param("status", "failed")
            mlflow.log_text(result.stderr, "error.log")
            return ExperimentRun(exp_id, config, run_id, "failed")
        
        # Parse results
        metrics = parse_results(result.stdout)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        
        mlflow.log_param("status", "completed")
        return ExperimentRun(exp_id, config, run_id, "completed", metrics)

def run_experiment_grid(exp_config_path: str) -> list[ExperimentRun]:
    """Run all combinations of independent variables."""
    with open(exp_config_path) as f:
        config = yaml.safe_load(f)
    
    # Generate all combinations
    independent_vars = config['experiment']['variables']['independent']
    param_grid = {var['name']: var['values'] for var in independent_vars}
    
    runs = []
    for combo in generate_combinations(param_grid):
        run_config = config.copy()
        # Update with this combination
        for k, v in combo.items():
            # Update in fixed_params or wherever needed
            pass
        
        run = run_experiment(run_config)
        runs.append(run)
    
    return runs
```

### 3. Results Analysis (`src/hardik_llm/experiments/analyze.py`)

```python
import pandas as pd
import matplotlib.pyplot as plt
import mlflow

def load_experiment_results(exp_id: str) -> pd.DataFrame:
    """Load all runs for an experiment from MLflow."""
    runs = mlflow.search_runs(
        experiment_names=["hardik-llm-lab"],
        filter_string=f"params.experiment_id = '{exp_id}'"
    )
    return runs

def plot_experiment_results(exp_id: str, x_var: str, y_vars: list[str]):
    """Generate standard plots for experiment."""
    df = load_experiment_results(exp_id)
    
    fig, axes = plt.subplots(len(y_vars), 1, figsize=(8, 4*len(y_vars)))
    if len(y_vars) == 1:
        axes = [axes]
    
    for ax, y_var in zip(axes, y_vars):
        ax.plot(df[f'params.{x_var}'], df[f'metrics.{y_var}'], 'o-')
        ax.set_xlabel(x_var)
        ax.set_ylabel(y_var)
        ax.set_title(f'{exp_id}: {y_var} vs {x_var}')
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'experiments/plots/{exp_id}_{x_var}_vs_{"_".join(y_vars)}.png')
    plt.close()

def generate_experiment_report(exp_id: str) -> str:
    """Generate markdown report for experiment."""
    df = load_experiment_results(exp_id)
    
    report = f"# {exp_id} Results\n\n"
    report += f"**Runs completed**: {len(df)}\n\n"
    report += "## Metrics Summary\n\n"
    report += df[[c for c in df.columns if c.startswith('metrics.')]].describe().to_markdown()
    report += "\n\n## Plots\n\n"
    for y_var in ['val_perplexity', 'compression_ratio', 'training_time_sec']:
        report += f"![{y_var}](plots/{exp_id}_vocab_size_vs_{y_var}.png)\n\n"
    
    return report
```

### 4. Experiment Registry (`docs/research/experiments.md`)

```markdown
# Experiment Registry

| ID | Question | Variable | Metric | Status | Result |
|----|----------|----------|--------|--------|--------|
| EXP-001 | Does vocab size affect training efficiency? | vocab_size | compression_ratio, val_perplexity | 🔄 Running | - |
| EXP-002 | Does head count affect quality? | n_heads | val_loss, throughput | ⏳ Planned | - |
| EXP-003 | How does context length affect memory? | max_seq_len | VRAM, throughput | ⏳ Planned | - |
| EXP-004 | Sinusoidal vs Learned PE? | pos_encoding | val_loss, extrapolation | ⏳ Planned | - |
| EXP-005 | Embedding dim vs quality? | d_model | val_loss, params | ⏳ Planned | - |
| EXP-006 | Weight tying effect? | tie_weights | val_perplexity | ⏳ Planned | - |
| EXP-007 | Head count vs quality? | n_heads | val_loss, throughput | ⏳ Planned | - |
| EXP-008 | Causal vs Full attention? | attention_type | val_perplexity | ⏳ Planned | - |
| EXP-009 | Manual vs PyTorch SDPA? | implementation | speed, memory, accuracy | ⏳ Planned | - |
| EXP-010 | Attention vs MLP-only? | architecture | val_loss | ⏳ Planned | - |
| EXP-011 | Pre-Norm vs Post-Norm? | norm_position | gradient_norm, depth | ⏳ Planned | - |
| EXP-012 | Model scaling laws? | model_size | val_perplexity | ⏳ Planned | - |
| EXP-013 | Weight tying effect? | tie_weights | val_perplexity | ⏳ Planned | - |
| EXP-014 | Gradient checkpointing? | checkpoint_freq | memory, speed | ⏳ Planned | - |
| EXP-015 | LR schedule comparison? | scheduler | val_loss, stability | ⏳ Planned | - |
| EXP-016 | Mixed precision effect? | precision | speed, memory, loss | ⏳ Planned | - |
| EXP-017 | Gradient accumulation? | accum_steps | speed, memory, loss | ⏳ Planned | - |
| EXP-018 | Weight decay values? | weight_decay | val_loss, overfitting | ⏳ Planned | - |
| EXP-019 | Freezing strategy? | freeze_strategy | accuracy, forgetting | ⏳ Planned | - |
| EXP-020 | Instruction data scaling? | dataset_size | quality, perplexity | ⏳ Planned | - |
| EXP-021 | Classification vs Instruction? | task_type | metrics | ⏳ Planned | - |
| EXP-022 | LoRA rank ablation? | rank | val_loss, params | ⏳ Planned | - |
| EXP-023 | LoRA vs Full FT? | method | val_loss, forgetting | ⏳ Planned | - |
| EXP-024 | LoRA target modules? | target_modules | val_loss | ⏳ Planned | - |
| EXP-025 | Alpha scaling? | alpha | stability, loss | ⏳ Planned | - |
| EXP-026 | Perplexity vs quality? | checkpoint | perplexity, human_score | ⏳ Planned | - |
| EXP-027 | Sampling params? | temp, top_k | quality, diversity | ⏳ Planned | - |
| EXP-028 | Few-shot evaluation? | n_shots | accuracy | ⏳ Planned | - |

**Status**: ⏳ Planned | 🔄 Running | ✅ Completed | ❌ Failed
```

---

## What to Observe & Note

### During Implementation
- [ ] MLflow tracking URI: local file vs server
- [ ] Config versioning: hash config for reproducibility
- [ ] Artifact storage: where to save plots, checkpoints
- [ ] Parallel runs: how to run multiple experiments concurrently

### During Experiments
- [ ] Every experiment gets an ID and entry in registry
- [ ] Results always saved to `experiments/results/EXP-XXX.json`
- [ ] Plots always saved to `experiments/plots/EXP-XXX/`
- [ ] Failed runs documented with error in `experiments/runs/EXP-XXX/error.log`

---

## Validation Tests (`tests/test_experiments.py`)

```python
def test_experiment_config_parsing():
    """YAML config loads correctly."""
    pass

def test_experiment_run_logging():
    """MLflow logs params, metrics, artifacts."""
    pass

def test_results_analysis():
    """Can load and plot results from MLflow."""
    pass
```

---

## Documentation to Write

1. `docs/research/experiments.md` - Master experiment registry
2. `docs/research/methodology.md` - How to design/run/analyze experiments
3. `guide/09-experiment-infra/EXPERIMENT_LOG.md`
4. Update `docs/decisions/ADR-004-why-mlflow.md`

---

## This Completes the Guide Directory

You now have 9 guides covering the full pipeline:

1. `guide/01-tokenization/` - BPE tokenizer + vocab experiments
2. `guide/02-embeddings/` - Token/positional embeddings
3. `guide/03-attention/` - Self-attention, multi-head, causal
4. `guide/04-transformer/` - Transformer blocks, GPT model
5. `guide/05-pretraining/` - Training pipeline, optimization
6. `guide/06-finetuning/` - Classification + instruction FT
7. `guide/07-lora/` - LoRA/PEFT implementation
8. `guide/08-evaluation/` - Perplexity, generation, benchmarks
9. `guide/09-experiment-infra/` - Experiment tracking, analysis

Each guide has:
- Implementation checklist (what to build)
- Experiments to run (with IDs, variables, metrics)
- What to observe (specific things to note)
- Validation tests (what to test)
- Documentation to write (where to record findings)

This is your **step-by-step laboratory manual**.