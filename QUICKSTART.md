# Quick Start Guide

## Installation

1. Install the package:
```bash
cd /root/ms
pip install -e .
```

2. (Optional) Install development dependencies:
```bash
pip install -e ".[dev]"
```

## Training

### Using Dummy Data (for testing)

The model comes with a dummy dataset generator for testing the architecture:

```bash
python scripts/train.py --config configs/default_config.yaml
```

### Using Real Data

1. Download/prepare your data:
```bash
python scripts/download_data.py --source example --output_dir data --num_samples 1000
```

2. Update the configuration file to point to your data:
```yaml
data:
  use_dummy_data: false
  train_data_path: data/train.csv
  val_data_path: data/val.csv
```

3. Train:
```bash
python scripts/train.py --config configs/default_config.yaml
```

## Inference

Predict spectrum for a peptide sequence:

```bash
python scripts/inference.py \
  --model_path checkpoints/ms_predictor_default_best.pt \
  --sequence PEPTIDE \
  --precursor_mz 500.0 \
  --charge 2 \
  --output predicted_spectrum.mgf
```

## Model Architecture

The model uses an Encoder-Decoder Transformer architecture:

1. **Encoder**: Processes peptide sequence (with metadata: precursor m/z, charge)
2. **Decoder**: Uses learnable query embeddings with bidirectional attention
3. **Prediction Heads**: Three heads predict m/z, intensity, and confidence

## Loss Functions

- **Hungarian Matching Loss**: Optimal bipartite matching between predictions and targets
- **Cosine Similarity Loss**: Global distribution consistency (tunable weight)

## Configuration

Key hyperparameters in `configs/default_config.yaml`:

- `model.hidden_dim`: Transformer hidden dimension (default: 512)
- `model.num_encoder_layers`: Number of encoder layers (default: 6)
- `model.num_decoder_layers`: Number of decoder layers (default: 6)
- `model.num_predictions`: Number of peaks to predict (default: 100)
- `loss.cosine_loss_weight`: Weight for cosine similarity loss (default: 0.5)

## Data Format

The dataset expects:
- Peptide amino acid sequences
- Precursor m/z values
- Charge states
- Fragment m/z arrays
- Fragment intensity arrays

See `src/ms_predictor/data/dataset.py` for implementation details.

## Tips

1. Start with dummy data to verify the model architecture works
2. Use mixed precision training (`training.mixed_precision: true`) for faster training
3. Adjust `loss.cosine_loss_weight` to balance individual peak matching vs. global distribution
4. Monitor both training and validation loss to detect overfitting
5. Use early stopping to avoid overtraining

## Troubleshooting

**Out of memory**: Reduce batch size or model size
**Slow training**: Enable mixed precision, increase batch size, or use more workers
**Poor predictions**: Increase model size, train longer, or tune loss weights

