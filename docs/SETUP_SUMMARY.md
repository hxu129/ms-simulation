# Environment Setup - Quick Reference

## Files Created

1. **`requirements.txt`** - Core dependencies
2. **`requirements-dev.txt`** - Development dependencies
3. **`ENVIRONMENT_SETUP.md`** - Comprehensive setup guide
4. **`setup_environment.sh`** - Automated setup script

## Quick Start

### Method 1: Automated Setup (Recommended)

```bash
# For CUDA 11.8 (most common)
bash setup_environment.sh 11.8

# For CUDA 12.1
bash setup_environment.sh 12.1

# For CPU only
bash setup_environment.sh cpu
```

### Method 2: Manual Setup with Conda

```bash
# Create environment
conda create -n ms-predictor python=3.9 -y
conda activate ms-predictor

# Install PyTorch (CUDA 11.8)
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

### Method 3: Manual Setup with venv

```bash
# Create environment
python3.9 -m venv venv
source venv/bin/activate

# Install PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

## Core Dependencies

```
torch>=2.0.0          # Deep learning framework
numpy>=1.24.0         # Numerical computing
pandas>=2.0.0         # Data manipulation
scipy>=1.10.0         # Scientific computing
pyarrow>=12.0.0       # Parquet support
h5py>=3.8.0          # HDF5 support (optional)
pyyaml>=6.0          # Configuration files
tqdm>=4.65.0         # Progress bars
```

## Verification

After setup, verify your installation:

```bash
# Test PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Test package
python -c "import ms_predictor; print('Package OK')"

# Run full test
python scripts/test_implementation.py

# Test OBS connection
python scripts/test_obs_connection.py
```

## System Requirements

**Recommended:**
- Ubuntu 20.04+ or similar Linux
- Python 3.9 or 3.10
- NVIDIA GPU with 8GB+ VRAM
- 16GB+ RAM
- 100GB+ SSD storage

**Minimum:**
- Any OS (Linux/Mac/Windows)
- Python 3.9+
- 8GB+ RAM
- 50GB+ storage
- (GPU optional, can run on CPU)

## Troubleshooting

**CUDA not detected:**
```bash
# Check NVIDIA driver
nvidia-smi

# Reinstall PyTorch with correct CUDA version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Import errors:**
```bash
# Reinstall package
pip install -e .
```

**Out of memory:**
```yaml
# In config file, reduce batch size
data:
  batch_size: 16  # or 8
```

## Next Steps

After environment setup:

1. **Download data:**
   ```bash
   python scripts/download_obs_data.py --num_files 5 --output_dir data
   ```

2. **Inspect data:**
   ```bash
   python scripts/download_obs_data.py --inspect data/MassIVE_KB_parquet/part_0001.parquet
   ```

3. **Update config:**
   ```bash
   # Edit configs/parquet_config.yaml
   # Set train_data_path to your data location
   ```

4. **Train model:**
   ```bash
   python scripts/train.py --config configs/parquet_config.yaml
   ```

## Complete Workflow

```bash
# 1. Setup environment
bash setup_environment.sh 11.8

# 2. Activate environment
conda activate ms-predictor

# 3. Download data (5 files for testing)
python scripts/download_obs_data.py --num_files 5

# 4. Train model
python scripts/train.py --config configs/parquet_config.yaml

# 5. Run inference
python scripts/inference.py \
  --model_path checkpoints/ms_predictor_parquet_best.pt \
  --sequence PEPTIDE \
  --precursor_mz 500.0 \
  --charge 2
```

## Documentation

- **ENVIRONMENT_SETUP.md** - Full setup guide with troubleshooting
- **PARQUET_DATA_GUIDE.md** - Data download and usage
- **QUICKSTART.md** - Quick start guide
- **README.md** - Project overview

All ready! 🚀

