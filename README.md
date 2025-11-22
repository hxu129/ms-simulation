# MS Spectrum Predictor

A Transformer-based model for predicting mass spectrometry spectra from peptide sequences.

## Architecture

- **Encoder-Decoder Transformer**: Processes peptide sequences and generates spectrum peaks
- **Set Prediction**: Uses Hungarian matching for optimal peak assignment
- **Loss Functions**: Combines Hungarian matching loss with cosine similarity for global distribution consistency

## Installation

```bash
pip install -e .
```

For development:
```bash
pip install -e ".[dev]"
```

## Project Structure

```
ms/
├── src/ms_predictor/        # Main package
│   ├── data/                # Data processing and tokenization
│   ├── model/               # Model architecture
│   ├── loss/                # Loss functions
│   ├── training/            # Training utilities
│   ├── inference/           # Inference pipeline
│   └── utils/               # Helper utilities
├── scripts/                 # Training and inference scripts
├── configs/                 # Configuration files
└── tests/                   # Unit tests
```

## Usage

The project uses [Hydra](https://hydra.cc/) for configuration management, enabling flexible command-line configuration overrides.

### Training

Basic training with default configuration:
```bash
python scripts/train.py
```

Override specific parameters:
```bash
python scripts/train.py data.batch_size=64 model.hidden_dim=1024
```

Use preset configurations:
```bash
# Small model for testing
python scripts/train.py +experiment=small

# Use Parquet data format
python scripts/train.py +data=parquet

# Combine multiple configs
python scripts/train.py +experiment=large +data=parquet
```

### Inference

```bash
python scripts/inference.py \
    model_path=checkpoints/best_model.pt \
    sequence=PEPTIDE \
    precursor_mz=500.5 \
    charge=2
```

Save output to file:
```bash
python scripts/inference.py \
    model_path=checkpoints/best_model.pt \
    sequence=PEPTIDE \
    precursor_mz=500.5 \
    charge=2 \
    output.file=prediction.mgf
```

For detailed configuration options, see [HYDRA_GUIDE.md](docs/HYDRA_GUIDE.md).

## Model Details

- **Input**: Peptide sequence + precursor m/z + charge state
- **Output**: Set of (m/z, intensity, confidence) triplets
- **Architecture**: Encoder-Decoder with bidirectional decoder attention
- **Loss**: Hungarian matching + cosine similarity (tunable)

## License

MIT

