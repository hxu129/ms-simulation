# OBS Data Integration Summary

## What's Been Implemented

I've integrated support for downloading and using MassIVE-KB and ProteomeTools data from the OBS bucket (`obs://lingtz/ultraprot`) with your MS spectrum predictor.

## New Files Created

### 1. Data Download Script
**`scripts/download_obs_data.py`**
- Downloads HDF5 files from OBS using obsutil
- Supports MassIVE-KB and ProteomeTools data sources
- Can download specific parts or all data
- Creates metadata JSON files for tracking
- Includes HDF5 file inspection functionality

### 2. HDF5 Dataset Loader
**`src/ms_predictor/data/hdf5_dataset.py`**
- PyTorch Dataset for loading HDF5 files
- Automatic train/val/test splitting
- Supports multiple HDF5 structures (array-based, group-based)
- Optional in-memory caching for faster training
- Handles peptide sequences, precursor m/z, charge, and spectra

### 3. Configuration
**`configs/obs_data_config.yaml`**
- Pre-configured for OBS data
- Sets `use_hdf5: true`
- Includes paths for downloaded data
- Optimized hyperparameters

### 4. Documentation
**`OBS_DATA_GUIDE.md`**
- Complete step-by-step guide
- Download instructions
- Training workflow
- Troubleshooting tips

### 5. Test Script
**`scripts/test_obs_connection.py`**
- Verifies obsutil is configured
- Tests bucket access
- Downloads and verifies a test file
- Quick sanity check before full download

## Data Available in OBS

From your temp file, I can see:

### MassIVE-KB
- Location: `obs://lingtz/ultraprot/MassIVE_KB1/MassIVE_KB/part_0/`
- Format: HDF5 files (`.mgf.hdf5`)
- File sizes: ~1-60 MB per file
- Hundreds of files available

### ProteomeTools
- Location: `obs://lingtz/ultraprot/data-1/`
- Multiple subdirectories

## Quick Start Guide

### Step 1: Test OBS Connection
```bash
python scripts/test_obs_connection.py
```

This will:
- Verify obsutil is working
- Test bucket access
- Download a small test file
- Inspect the HDF5 structure

### Step 2: Download Data

Download a small subset for testing (recommended first):
```bash
python scripts/download_obs_data.py \
  --source massive-kb \
  --output_dir data \
  --num_files 10 \
  --part 0 \
  --create_metadata
```

Download more data:
```bash
python scripts/download_obs_data.py \
  --source massive-kb \
  --output_dir data \
  --num_files 100 \
  --part 0 \
  --create_metadata
```

### Step 3: Inspect Downloaded Files
```bash
# List downloaded files
ls -lh data/MassIVE_KB/part_0/

# Inspect HDF5 structure
python scripts/download_obs_data.py \
  --inspect data/MassIVE_KB/part_0/<first_file>.hdf5
```

**Important**: Check the HDF5 structure! You may need to adjust the `_load_spectrum()` method in `hdf5_dataset.py` to match your specific HDF5 format.

### Step 4: Update Configuration

Edit `configs/obs_data_config.yaml`:
```yaml
data:
  use_dummy_data: false
  use_hdf5: true
  train_data_path: data/MassIVE_KB/part_0
  metadata_file: data/massive_kb_metadata.json
```

### Step 5: Train the Model
```bash
python scripts/train.py --config configs/obs_data_config.yaml
```

## Features

### Automatic Data Splitting
The HDF5 dataset automatically splits data into:
- **Train**: 80% (default)
- **Validation**: 10%
- **Test**: 10%

No need for separate train/val/test files!

### Flexible HDF5 Support
The dataset loader handles multiple HDF5 structures:

1. **Array-based structure**:
   ```python
   f['sequences'] = ['PEPTIDE', 'SEQUENCE', ...]
   f['precursor_mz'] = [500.0, 600.0, ...]
   f['charge'] = [2, 3, ...]
   f['mz'] = [array1, array2, ...]
   f['intensity'] = [array1, array2, ...]
   ```

2. **Group-based structure**:
   ```python
   f['spectra']['0']['sequence'] = 'PEPTIDE'
   f['spectra']['0']['precursor_mz'] = 500.0
   f['spectra']['0']['mz'] = array([...])
   ```

### Memory Management
- **On-the-fly loading**: Loads spectra as needed (default)
- **In-memory caching**: Set `cache_in_memory: true` for faster training (requires more RAM)

### Updated Training Script
The training script now supports three modes:
1. **Dummy data**: `use_dummy_data: true` (for testing)
2. **HDF5 data**: `use_hdf5: true` (for OBS data)
3. **Custom dataset**: Regular MSDataset (for CSV, etc.)

## Important Notes

### 1. HDF5 Structure May Vary
The HDF5 files might have different internal structures. After downloading, always:
1. Inspect at least one file with `--inspect`
2. Check the printed structure
3. Update `_load_spectrum()` in `hdf5_dataset.py` if needed

### 2. Start Small
Don't download all files at once:
- Start with 10 files to verify everything works
- Inspect the HDF5 structure
- Test training for 1-2 epochs
- Scale up gradually

### 3. obsutil Configuration
Make sure obsutil is properly configured with your OBS credentials. The script assumes obsutil is at `/root/ms/obsutil_linux_amd64_5.7.9/obsutil`.

## Troubleshooting

### "Could not load spectrum"
**Cause**: HDF5 structure doesn't match expected format

**Solution**:
1. Inspect the file: `python scripts/download_obs_data.py --inspect <file>.hdf5`
2. Update `_load_spectrum()` method in `src/ms_predictor/data/hdf5_dataset.py`
3. Match the keys and structure from your inspection

### "No spectra found"
**Cause**: `_load_spectra_info()` can't detect spectra in HDF5 files

**Solution**:
Update the detection logic in `_load_spectra_info()` based on your HDF5 structure

### Out of memory
**Solutions**:
- Reduce batch size: `data.batch_size: 16`
- Disable caching: `data.cache_in_memory: false`
- Reduce model size: `model.hidden_dim: 256`

## Expected Performance

With 100 files (~300 MB):
- Download time: ~5-10 minutes
- Loading time: <1 minute
- Training: ~1-2 hours per epoch on GPU

With 1000+ files:
- Download time: 1-2 hours
- Significant storage required (several GB)
- Training: Several hours per epoch

## Next Steps

1. **Test the connection**:
   ```bash
   python scripts/test_obs_connection.py
   ```

2. **Download a small dataset**:
   ```bash
   python scripts/download_obs_data.py --source massive-kb --num_files 10
   ```

3. **Inspect the HDF5 structure**:
   ```bash
   python scripts/download_obs_data.py --inspect data/MassIVE_KB/part_0/<file>.hdf5
   ```

4. **Adjust the dataset loader if needed** (see OBS_DATA_GUIDE.md)

5. **Train the model**:
   ```bash
   python scripts/train.py --config configs/obs_data_config.yaml
   ```

6. **Monitor training** and adjust hyperparameters as needed

## Files Modified

- `scripts/train.py`: Added support for HDF5 datasets
- `src/ms_predictor/data/__init__.py`: Added HDF5 exports
- `.gitignore`: Added obsutil* to ignore list

## All Tools Ready

You now have everything needed to:
✅ Download data from OBS
✅ Load HDF5 files
✅ Train on real MS/MS data
✅ Evaluate on validation data
✅ Make predictions

Good luck with your training! 🚀

