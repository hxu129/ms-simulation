# MS Spectrum Predictor Environment Setup Guide

This guide explains how to set up your Python environment for the MS spectrum predictor.

## Option 1: Using Conda (Recommended)

### Step 1: Create Conda Environment

```bash
# Create new environment with Python 3.9+
conda create -n ms-predictor python=3.9 -y
conda activate ms-predictor
```

### Step 2: Install PyTorch (Choose based on your system)

**For CUDA 11.8** (Most common for recent GPUs):
```bash
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

**For CUDA 12.1** (Latest GPUs):
```bash
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

**For CPU only** (No GPU):
```bash
conda install pytorch torchvision cpuonly -c pytorch -y
```

### Step 3: Install Other Dependencies

```bash
# Install from requirements.txt
pip install -r requirements.txt
```

### Step 4: Install the Package

```bash
# Install in development mode
pip install -e .
```

### Step 5: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python scripts/test_implementation.py
```

---

## Option 2: Using pip + venv

### Step 1: Create Virtual Environment

```bash
# Create virtual environment
python3.9 -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
# venv\Scripts\activate
```

### Step 2: Upgrade pip

```bash
pip install --upgrade pip setuptools wheel
```

### Step 3: Install PyTorch

Visit https://pytorch.org/get-started/locally/ and get the appropriate command for your system.

Example for CUDA 11.8:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Step 4: Install Other Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Install the Package

```bash
pip install -e .
```

---

## Option 3: Quick Install (All-in-one)

If you have conda and want a quick setup:

```bash
# Create and activate environment
conda create -n ms-predictor python=3.9 -y
conda activate ms-predictor

# Install PyTorch with CUDA 11.8
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install everything else
pip install -r requirements.txt
pip install -e .

# Verify
python scripts/test_implementation.py
```

---

## Development Setup

If you're developing the code, install development dependencies:

```bash
pip install -r requirements-dev.txt
```

This includes:
- Testing tools (pytest)
- Code formatters (black, isort)
- Linters (flake8)
- Documentation tools (sphinx)

---

## Verifying Your Installation

### Quick Test
```bash
python -c "import ms_predictor; print('✓ Package imported successfully')"
```

### Full Test
```bash
python scripts/test_implementation.py
```

### Check GPU
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

### Test OBS Connection
```bash
python scripts/test_obs_connection.py
```

---

## Common Issues & Solutions

### Issue 1: CUDA Version Mismatch

**Error**: "RuntimeError: CUDA error: no kernel image is available for execution"

**Solution**: Reinstall PyTorch with the correct CUDA version for your system:
```bash
# Check your CUDA version
nvidia-smi

# Install matching PyTorch version
# See https://pytorch.org/get-started/locally/
```

### Issue 2: Out of Memory

**Error**: "CUDA out of memory"

**Solution**: Reduce batch size in config:
```yaml
data:
  batch_size: 16  # or even 8
```

### Issue 3: Slow Installation

**Solution**: Use conda for faster installation:
```bash
conda install -c conda-forge numpy pandas scipy tqdm pyyaml pyarrow h5py -y
```

### Issue 4: Import Errors

**Error**: "ModuleNotFoundError: No module named 'ms_predictor'"

**Solution**: Install the package in development mode:
```bash
cd /root/ms
pip install -e .
```

---

## Environment Variables (Optional)

You can set these for better performance:

```bash
# Enable TF32 for faster training on Ampere GPUs
export NVIDIA_TF32_OVERRIDE=1

# Reduce memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Set number of threads
export OMP_NUM_THREADS=4
```

Add to your `.bashrc` or `.zshrc` for persistence.

---

## Recommended Environment

For best performance:
- **OS**: Linux (Ubuntu 20.04+ or similar)
- **Python**: 3.9 or 3.10
- **PyTorch**: 2.0+ with CUDA 11.8 or 12.1
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3060 or better)
- **RAM**: 16GB+ system memory
- **Storage**: SSD with 100GB+ free space

Minimum requirements:
- **OS**: Any (Linux/Mac/Windows)
- **Python**: 3.9+
- **PyTorch**: 2.0+ (CPU version works)
- **RAM**: 8GB+
- **Storage**: 50GB+

---

## Docker Setup (Alternative)

If you prefer Docker:

```dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime

WORKDIR /workspace
COPY . /workspace/

RUN pip install -r requirements.txt
RUN pip install -e .

CMD ["/bin/bash"]
```

Build and run:
```bash
docker build -t ms-predictor .
docker run --gpus all -it ms-predictor
```

---

## Updating Dependencies

To update all packages to their latest versions:

```bash
# Update pip packages
pip install --upgrade -r requirements.txt

# Update conda packages (if using conda)
conda update --all -y
```

---

## Troubleshooting

If you encounter any issues:

1. **Check Python version**: `python --version` (should be 3.9+)
2. **Check PyTorch installation**: `python -c "import torch; print(torch.__version__)"`
3. **Check CUDA**: `python -c "import torch; print(torch.cuda.is_available())"`
4. **Reinstall from scratch**: Delete environment and start over
5. **Check GitHub Issues**: See if others had similar problems

For more help, create an issue with:
- Your OS and Python version
- Output of `pip list`
- Full error message

---

## Quick Reference

```bash
# Activate environment
conda activate ms-predictor

# Deactivate
conda deactivate

# List installed packages
pip list

# Update a specific package
pip install --upgrade package-name

# Remove environment (conda)
conda env remove -n ms-predictor

# Remove environment (venv)
rm -rf venv/
```

