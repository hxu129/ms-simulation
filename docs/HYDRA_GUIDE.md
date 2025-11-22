# Hydra Configuration Guide

This guide explains how to use the Hydra configuration system for training and inference with the MS spectrum predictor.

## Overview

The project now uses [Hydra](https://hydra.cc/) for configuration management, which provides:
- **Command-line overrides**: Change any configuration parameter without editing files
- **Config composition**: Combine multiple configuration files
- **Automatic output management**: Each run gets its own output directory
- **Config versioning**: Every run saves its complete configuration

## Installation

Hydra is included in `requirements.txt`. To install:

```bash
pip install -r requirements.txt
```

Or specifically:

```bash
pip install hydra-core>=1.3.0
```

## Configuration Structure

```
configs/
├── config.yaml              # Main Hydra entry point
├── default_config.yaml      # Default configuration values
├── inference.yaml           # Inference-specific configuration
├── experiment/              # Experiment presets
│   ├── small.yaml          # Small model for testing
│   └── large.yaml          # Large model for production
└── data/                    # Data source configurations
    ├── dummy.yaml          # Dummy data for testing
    ├── hdf5.yaml           # HDF5 data format
    └── parquet.yaml        # Parquet data format (recommended)
```

## Training Usage

### 1. Basic Training (Default Configuration)

```bash
python scripts/train.py
```

This uses all default settings from `configs/default_config.yaml`.

### 2. Override Specific Parameters

```bash
# Override single parameters
python scripts/train.py data.batch_size=64

# Override multiple parameters
python scripts/train.py data.batch_size=64 model.hidden_dim=1024 training.num_epochs=200

# Override nested parameters
python scripts/train.py optimizer.learning_rate=0.0001 optimizer.warmup_epochs=10
```

### 3. Use Config Group Variants

Config groups allow you to switch entire configuration sections:

```bash
# Use small model configuration
python scripts/train.py +experiment=small

# Use parquet data configuration
python scripts/train.py +data=parquet

# Combine multiple config groups
python scripts/train.py +experiment=large +data=parquet
```

**Note**: Use the `+` prefix to add config groups that aren't in the defaults list.

### 4. Combine Groups and Overrides

```bash
# Start with small model but customize parameters
python scripts/train.py +experiment=small data.batch_size=128 training.num_epochs=50

# Use parquet data with custom path
python scripts/train.py +data=parquet data.train_data_path=/path/to/data
```

### 5. Resume Training

```bash
python scripts/train.py +resume_path=checkpoints/best_model.pt
```

### 6. Advanced: Multi-run Sweeps

Run multiple experiments with different parameters:

```bash
python scripts/train.py --multirun \
    model.hidden_dim=256,512,1024 \
    data.batch_size=32,64
```

This runs 6 experiments (3 hidden_dim × 2 batch_size combinations).

## Inference Usage

### 1. Basic Inference

```bash
python scripts/inference.py \
    model_path=checkpoints/best_model.pt \
    sequence=PEPTIDE \
    precursor_mz=500.5 \
    charge=2
```

### 2. Save Output to File

```bash
python scripts/inference.py \
    model_path=checkpoints/best_model.pt \
    sequence=PEPTIDE \
    precursor_mz=500.5 \
    charge=2 \
    output.file=prediction.mgf
```

### 3. Override Inference Settings

```bash
python scripts/inference.py \
    model_path=checkpoints/best_model.pt \
    sequence=PEPTIDE \
    precursor_mz=500.5 \
    charge=2 \
    inference.confidence_threshold=0.7 \
    inference.max_intensity=10000.0
```

## Output Directory Management

By default, Hydra creates a new output directory for each run:

```
outputs/
└── ms_predictor_default/
    └── 2024-01-15_10-30-00/
        ├── .hydra/
        │   ├── config.yaml          # Complete resolved config
        │   ├── hydra.yaml           # Hydra runtime config
        │   └── overrides.yaml       # Command-line overrides
        ├── train.log                # Training logs (if configured)
        └── checkpoints/             # Model checkpoints
```

### Change Output Directory

```bash
# Custom output directory
python scripts/train.py hydra.run.dir=my_experiment/run_001

# Disable directory creation (use current directory)
python scripts/train.py hydra.run.dir=.
```

## Configuration Examples

### Example 1: Quick Testing

```bash
# Small model, few epochs, dummy data
python scripts/train.py \
    +experiment=small \
    +data=dummy \
    training.num_epochs=5 \
    training.log_interval=5
```

### Example 2: Production Training

```bash
# Large model with real data
python scripts/train.py \
    +experiment=large \
    +data=parquet \
    data.train_data_path=/data/obs/parquet_files \
    training.num_epochs=200 \
    training.checkpoint_dir=checkpoints/production
```

### Example 3: Hyperparameter Tuning

```bash
# Try different learning rates
python scripts/train.py --multirun \
    +experiment=small \
    optimizer.learning_rate=0.0001,0.0005,0.001 \
    optimizer.warmup_epochs=5,10
```

## Viewing Configuration

### See Current Configuration

```bash
# Show complete resolved configuration
python scripts/train.py --cfg job

# Show only config groups available
python scripts/train.py --help
```

### Print Configuration in Code

The training script automatically prints the full configuration at the start of each run.

## Creating Custom Config Groups

### Add a New Experiment Configuration

Create `configs/experiment/my_experiment.yaml`:

```yaml
# @package _global_

experiment_name: my_custom_experiment

model:
  hidden_dim: 768
  num_encoder_layers: 8

training:
  num_epochs: 150
  learning_rate: 0.0002
```

Use it:

```bash
python scripts/train.py +experiment=my_experiment
```

### Add a New Data Configuration

Create `configs/data/my_data.yaml`:

```yaml
# @package _global_

data:
  train_data_path: /path/to/my/data
  use_parquet: true
  batch_size: 64
  cache_dataframes: true
```

Use it:

```bash
python scripts/train.py +data=my_data
```

## Tips and Best Practices

1. **Start with defaults**: Run with default config first, then override what you need
2. **Use config groups**: Create reusable config groups for common experiment setups
3. **Track experiments**: Each run's config is saved in `.hydra/config.yaml`
4. **Compose configs**: Combine experiment + data configs for maximum flexibility
5. **Use sweeps**: Automate hyperparameter searches with `--multirun`

## Troubleshooting

### "Could not override 'experiment'"

Use `+experiment=small` (with `+`) to add config groups not in the defaults list.

### "Working directory changed"

Hydra changes the working directory to the output directory. Use `hydra.utils.get_original_cwd()` to get the original directory (this is already handled in the training script).

### "Config file not found"

Ensure you're running from the project root directory, or adjust `config_path` in the `@hydra.main` decorator.

## Migration from Old Configuration

If you have old scripts using the argparse-based configuration:

**Old way:**
```bash
python scripts/train.py --config configs/my_config.yaml
```

**New way:**
```bash
python scripts/train.py --config-name my_config
# or just override parameters:
python scripts/train.py data.batch_size=64
```

The old `Config.from_yaml()` method still works for backward compatibility, but it's recommended to use Hydra for new scripts.

## Learn More

- [Hydra Documentation](https://hydra.cc/)
- [Hydra Tutorials](https://hydra.cc/docs/tutorials/intro/)
- [OmegaConf Documentation](https://omegaconf.readthedocs.io/)


