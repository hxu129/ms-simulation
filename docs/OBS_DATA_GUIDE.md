# Using OBS Data with MS Predictor

This guide explains how to download and use MassIVE-KB and ProteomeTools data from the OBS bucket for training the MS spectrum predictor.

## Data Sources

The data is stored in Huawei OBS at `obs://lingtz/ultraprot/`:

1. **MassIVE-KB**: `obs://lingtz/ultraprot/MassIVE_KB1/MassIVE_KB/part_0/`
   - Contains HDF5 files with MS/MS spectra
   - Large-scale spectral library

2. **ProteomeTools**: `obs://lingtz/ultraprot/data-1/`
   - Synthetic peptide spectra
   - High-quality reference data

## Step 1: Download Data

### Download MassIVE-KB Data

Download a subset of files for testing (e.g., 10 files):

```bash
python scripts/download_obs_data.py \
  --source massive-kb \
  --output_dir data \
  --num_files 10 \
  --part 0 \
  --create_metadata
```

Download all files (this will take time):

```bash
python scripts/download_obs_data.py \
  --source massive-kb \
  --output_dir data \
  --part 0 \
  --create_metadata
```

### Download ProteomeTools Data

```bash
python scripts/download_obs_data.py \
  --source proteometools \
  --output_dir data \
  --num_files 10 \
  --create_metadata
```

### Download Both

```bash
python scripts/download_obs_data.py \
  --source both \
  --output_dir data \
  --num_files 10 \
  --create_metadata
```

## Step 2: Inspect HDF5 Files

Before training, inspect the structure of downloaded HDF5 files:

```bash
python scripts/download_obs_data.py \
  --inspect data/MassIVE_KB/part_0/<filename>.hdf5
```

This will show:
- Dataset structure
- Data shapes and types
- Available fields

## Step 3: Update HDF5 Dataset Loader (if needed)

The HDF5 dataset loader in `src/ms_predictor/data/hdf5_dataset.py` attempts to handle common HDF5 formats. If your data has a different structure, you may need to update the `_load_spectrum()` method to match your HDF5 structure.

Common HDF5 structures:

### Structure 1: Array-based
```python
f['sequences'] = ['PEPTIDE', 'SEQUENCE', ...]
f['precursor_mz'] = [500.0, 600.0, ...]
f['charge'] = [2, 3, ...]
f['mz'] = [array1, array2, ...]
f['intensity'] = [array1, array2, ...]
```

### Structure 2: Group-based
```python
f['spectra']['0']['sequence'] = 'PEPTIDE'
f['spectra']['0']['precursor_mz'] = 500.0
f['spectra']['0']['charge'] = 2
f['spectra']['0']['mz'] = array([...])
f['spectra']['0']['intensity'] = array([...])
```

## Step 4: Update Configuration

Edit `configs/obs_data_config.yaml`:

```yaml
data:
  use_dummy_data: false
  use_hdf5: true
  train_data_path: data/MassIVE_KB/part_0  # Path to downloaded data
  metadata_file: data/massive_kb_metadata.json
  cache_in_memory: false  # Set to true if you have enough RAM
```

## Step 5: Train the Model

```bash
python scripts/train.py --config configs/obs_data_config.yaml
```

### Training Tips

1. **Start small**: Begin with 10-100 files to verify everything works
2. **Monitor memory**: HDF5 files are loaded on-the-fly, but you can cache in memory if you have enough RAM
3. **Adjust batch size**: Reduce if you run out of GPU memory
4. **Use mixed precision**: Enabled by default for faster training

## Step 6: Monitor Training

The trainer will:
- Log progress every 10 batches
- Validate after each epoch
- Save checkpoints every 5 epochs
- Save the best model based on validation loss
- Apply early stopping if validation loss doesn't improve

Checkpoints are saved to `checkpoints/`:
- `ms_predictor_obs_data_best.pt`: Best model
- `ms_predictor_obs_data_epoch_N.pt`: Periodic checkpoints

## Step 7: Inference

After training, use the best model for predictions:

```bash
python scripts/inference.py \
  --model_path checkpoints/ms_predictor_obs_data_best.pt \
  --sequence PEPTIDE \
  --precursor_mz 500.0 \
  --charge 2 \
  --output predicted_spectrum.mgf
```

## Troubleshooting

### Issue: "Could not load HDF5 file"

**Solution**: Inspect the HDF5 structure and update `_load_spectrum()` in `hdf5_dataset.py`:

```bash
python scripts/download_obs_data.py --inspect data/MassIVE_KB/part_0/<file>.hdf5
```

### Issue: Out of memory during training

**Solutions**:
1. Reduce batch size: `data.batch_size: 16`
2. Reduce model size: `model.hidden_dim: 256`
3. Disable caching: `data.cache_in_memory: false`
4. Use gradient accumulation (modify trainer)

### Issue: Training is slow

**Solutions**:
1. Enable mixed precision (already enabled by default)
2. Increase batch size if memory allows
3. Use more data workers: `data.num_workers: 8`
4. Cache data in memory if you have enough RAM: `data.cache_in_memory: true`

### Issue: Dataset is empty

**Cause**: HDF5 structure doesn't match expected format

**Solution**: 
1. Inspect HDF5 file structure
2. Update `_load_spectra_info()` and `_load_spectrum()` methods

## Data Statistics

After downloading, you can check the metadata file for statistics:

```bash
cat data/massive_kb_metadata.json
```

This shows:
- Number of files
- Total size
- File locations

## Advanced Usage

### Train on Multiple Parts

MassIVE-KB is split into multiple parts. To train on multiple parts:

1. Download multiple parts:
```bash
for part in 0 1 2; do
  python scripts/download_obs_data.py \
    --source massive-kb \
    --output_dir data \
    --part $part \
    --num_files 10
done
```

2. Update config to point to all parts (you may need to modify the dataset loader to support multiple directories)

### Combine MassIVE-KB and ProteomeTools

Download both datasets and create a combined dataset class that loads from both sources.

## Performance Expectations

With 10,000 spectra:
- Training time: ~1-2 hours per epoch (on GPU)
- Model size: ~50-100 MB
- Memory usage: ~4-8 GB GPU RAM (batch size 32)

With full MassIVE-KB:
- Training time: Several days
- Requires significant storage and compute resources

## Next Steps

1. Start with a small dataset (10-100 files)
2. Verify the data loader works correctly
3. Train for a few epochs to check convergence
4. Scale up to full dataset
5. Tune hyperparameters based on validation performance

