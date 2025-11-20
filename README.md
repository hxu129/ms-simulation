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

### Training

```bash
python scripts/train.py --config configs/default_config.yaml
```

### Inference

```bash
python scripts/inference.py --model_path checkpoints/model.pt --sequence PEPTIDE
```

## Model Details

- **Input**: Peptide sequence + precursor m/z + charge state
- **Output**: Set of (m/z, intensity, confidence) triplets
- **Architecture**: Encoder-Decoder with bidirectional decoder attention
- **Loss**: Hungarian matching + cosine similarity (tunable)

## License

MIT

