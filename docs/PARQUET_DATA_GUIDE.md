# Using Parquet Data from OBS

This guide explains how to download and use MassIVE-KB data in **Parquet format** from the OBS bucket. Parquet files are more efficient and easier to work with than HDF5 files.

## Why Parquet?

- **Efficient**: Columnar storage format, faster to read
- **Large files**: Each file contains ~150-240 MB of spectra (thousands of spectra per file)
- **Easy to inspect**: Can be opened with pandas directly
- **Better compression**: Smaller total size than HDF5

## Data Available

**MassIVE-KB (Parquet)**:
- Location: `obs://lingtz/ultraprot/data-1/MassIVE_KB/`
- Format: Parquet files (`part_XXXX.parquet`)
- File sizes: 150-240 MB each
- Thousands of spectra per file
- Available parts: 0001, 0005, 0025, 0043, 0078, 0082-0134, and more

## Quick Start (3 Steps)

### Step 1: Download Data

Download to default location (`data/`):
```bash
# Download 5 files for testing
python scripts/download_obs_data.py --num_files 5

# Download specific parts
python scripts/download_obs_data.py --parts 1 5 25

# Download to custom location
python scripts/download_obs_data.py --num_files 10 --output_dir /mnt/storage/ms_data
```

**Command Options**:
- `--output_dir PATH`: Where to save downloaded data (default: `data/`)
- `--num_files N`: Download first N files
- `--parts N1 N2 ...`: Download specific part numbers (e.g., `--parts 1 5 25 43`)
- `--create_metadata`: Create a metadata JSON file with statistics
- `--inspect_first`: Inspect the first downloaded file automatically
- `--force`: Force redownload even if files exist

### Step 2: Inspect Data

Inspect a downloaded Parquet file:
```bash
python scripts/download_obs_data.py --inspect data/MassIVE_KB_parquet/part_0001.parquet
```

This shows:
- Number of spectra
- Column names
- Data types
- Example rows
- Sample spectrum data

### Step 3: Update Configuration

Edit `configs/parquet_config.yaml`:
```yaml
data:
  use_dummy_data: false
  use_parquet: true
  train_data_path: data/MassIVE_KB_parquet  # Or your custom location
  metadata_file: data/massive_kb_metadata.json  # Optional
```

### Step 4: Train the Model

```bash
python scripts/train.py --config configs/parquet_config.yaml
```

## Expected Parquet Structure

The Parquet files should contain these columns:
- `sequence`: Peptide amino acid sequence (string)
- `precursor_mz`: Precursor m/z value (float)
- `charge`: Charge state (int)
- `mz`: Array of fragment m/z values (list/array)
- `intensity`: Array of fragment intensities (list/array)

Alternative column names supported:
- `prec_mz` instead of `precursor_mz`
- `prec_charge` instead of `charge`
- `mz_array` instead of `mz`
- `int_array` or `intensities` instead of `intensity`

## Examples

### Example 1: Quick Test with 5 Files

```bash
# Download 5 files to test
python scripts/download_obs_data.py \
  --num_files 5 \
  --inspect_first \
  --create_metadata

# Check what was downloaded
ls -lh data/MassIVE_KB_parquet/

# Train
python scripts/train.py --config configs/parquet_config.yaml
```

### Example 2: Custom Storage Location

```bash
# Download to external drive
python scripts/download_obs_data.py \
  --num_files 20 \
  --output_dir /mnt/external/ms_data \
  --create_metadata

# Update config
# Edit configs/parquet_config.yaml:
#   data.train_data_path: /mnt/external/ms_data/MassIVE_KB_parquet

# Train
python scripts/train.py --config configs/parquet_config.yaml
```

### Example 3: Specific Parts

```bash
# Download specific high-quality parts
python scripts/download_obs_data.py \
  --parts 1 5 25 43 78 \
  --output_dir data \
  --create_metadata

# This downloads: part_0001, part_0005, part_0025, part_0043, part_0078
```

### Example 4: Full Dataset

```bash
# Download all available Parquet files (this will take time!)
python scripts/download_obs_data.py \
  --output_dir /root/autodl-tmp \
  --create_metadata

# This may download 50+ GB of data
```

## Configuration Options

In `configs/parquet_config.yaml`:

```yaml
data:
  train_data_path: data/MassIVE_KB_parquet  # Where you downloaded data
  metadata_file: data/massive_kb_metadata.json  # Optional
  
  cache_dataframes: false  # Set true to cache entire Parquet files in RAM
  max_files: null  # Limit number of files (e.g., 10 for testing)
  
  batch_size: 32  # Adjust based on GPU memory
  num_workers: 4  # Number of data loading threads
```

**Performance Tips**:
- `cache_dataframes: true` - Much faster but requires more RAM (~200 MB per file)
- `max_files: 10` - Use during development to iterate faster
- `num_workers: 8` - Increase if you have many CPU cores

## Data Statistics

After downloading with `--create_metadata`, check the metadata file:

```bash
cat data/massive_kb_metadata.json
```

Example output:
```json
{
  "data_dir": "data/MassIVE_KB_parquet",
  "num_files": 5,
  "total_size_mb": 1050.5,
  "total_spectra": 125000,
  "files": [...]
}
```

## Troubleshooting

### Issue: "Column 'sequence' not found"

**Cause**: Parquet file has different column names

**Solution**: Inspect the file to see actual columns:
```bash
python scripts/download_obs_data.py --inspect data/MassIVE_KB_parquet/part_0001.parquet
```

Then update `_load_spectrum()` in `src/ms_predictor/data/parquet_dataset.py` to match your column names.

### Issue: Out of memory

**Solutions**:
- Set `cache_dataframes: false`
- Reduce batch size: `batch_size: 16`
- Reduce `max_files`: `max_files: 5`
- Use fewer data workers: `num_workers: 2`

### Issue: Training is slow

**Solutions**:
- Enable caching: `cache_dataframes: true` (if you have enough RAM)
- Increase batch size: `batch_size: 64`
- More workers: `num_workers: 8`
- Use SSD storage for better I/O

### Issue: "obsutil: command not found"

**Solution**: Update the obsutil path in `scripts/download_obs_data.py`:
```python
OBSUTIL_PATH = "/path/to/your/obsutil"
```

## Advantages over HDF5

| Feature | Parquet | HDF5 |
|---------|---------|------|
| File size | 150-240 MB | 1-60 MB |
| Spectra per file | Thousands | Hundreds |
| Easy to inspect | ✅ (pandas) | ❌ (h5py) |
| Download time | Longer per file | Faster per file |
| Total files needed | Fewer | More |
| Pandas support | ✅ Native | ❌ Needs conversion |

## Performance Expectations

**With 5 files (~1 GB, ~25,000 spectra)**:
- Download time: 5-10 minutes
- Training: ~30 minutes per epoch (GPU)
- Memory: ~4-6 GB GPU RAM

**With 20 files (~4 GB, ~100,000 spectra)**:
- Download time: 20-30 minutes
- Training: ~2 hours per epoch (GPU)
- Memory: ~6-8 GB GPU RAM

**With all files (50+ GB, 1M+ spectra)**:
- Download time: 2-4 hours
- Training: Several days
- Requires significant compute resources

## Next Steps

1. **Start small**: Download 5 files to verify everything works
2. **Inspect**: Check the Parquet structure
3. **Test train**: Train for 1-2 epochs
4. **Scale up**: Download more files gradually
5. **Tune**: Adjust hyperparameters based on validation performance

## Comparison with Previous Methods

| Method | Data Format | Ease of Use | Recommended |
|--------|-------------|-------------|-------------|
| **Parquet** | ✅ Columnar | ✅ Easy | ✅ **YES** |
| HDF5 | Binary | ⚠️ Moderate | ⚠️ If needed |
| MGF | Text | ❌ Complex | ❌ No |

**Recommendation**: Use Parquet format (this guide) for the best experience!

