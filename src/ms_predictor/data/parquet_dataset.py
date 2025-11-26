"""
Parquet Dataset loader for MassIVE-KB and ProteomeTools data.

This is the only supported data loading format for the MS predictor.
"""

import os
import torch
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path

from .tokenizer import AminoAcidTokenizer
from .preprocessing import SpectrumPreprocessor


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.
    
    Args:
        batch: List of samples from dataset
        
    Returns:
        Batched dictionary
    """
    if len(batch) == 0:
        return {}
    
    # Pre-allocate result dict
    keys = batch[0].keys()
    result = {}
    
    # Stack each field efficiently
    for key in keys:
        # Direct stack - PyTorch optimizes this internally
        result[key] = torch.stack([sample[key] for sample in batch], dim=0)
    
    return result


class ParquetMSDataset(Dataset):
    """
    Dataset for loading MS/MS data from Parquet files.
    
    Supports MassIVE-KB and ProteomeTools data in Parquet format.
    """
    
    def __init__(
        self,
        data_dir: str = None,
        metadata_file: Optional[str] = None,
        tokenizer: Optional[AminoAcidTokenizer] = None,
        preprocessor: Optional[SpectrumPreprocessor] = None,
        max_length: int = 50,
        split: str = 'train',
        val_split: float = 0.1,
        test_split: float = 0.1,
        cache_dataframes: bool = False,
        max_files: Optional[int] = None,
        max_mz: float = 2000.0,
        top_k: int = 200,
        num_predictions: int = 100,
        shared_spectra_info: Optional[Tuple] = None,
        parquet_files: Optional[List[str]] = None
    ):
        """
        Initialize Parquet dataset.
        
        Args:
            data_dir: Directory containing Parquet files
            metadata_file: Optional JSON metadata file listing all Parquet files
            tokenizer: Amino acid tokenizer
            preprocessor: Spectrum preprocessor
            max_length: Maximum sequence length
            split: Dataset split ('train', 'val', 'test')
            val_split: Fraction of data for validation
            test_split: Fraction of data for testing
            cache_dataframes: Whether to cache loaded DataFrames in memory
            max_files: Maximum number of Parquet files to load (None for all)
            max_mz: Maximum m/z value for normalization
            top_k: Number of top peaks to extract from spectrum
            num_predictions: Number of predictions the model will make (N)
            shared_spectra_info: Optional pre-loaded spectra info (file_indices, row_indices, file_paths)
            parquet_files: Optional pre-loaded list of parquet file paths
        """
        self.data_dir = data_dir
        self.tokenizer = tokenizer or AminoAcidTokenizer()
        self.preprocessor = preprocessor or SpectrumPreprocessor(
            max_mz=max_mz,
            top_k=top_k,
            num_predictions=num_predictions
        )
        self.max_length = max_length
        self.split = split
        self.cache_dataframes = cache_dataframes
        
        # IMPORTANT: Initialize df_cache BEFORE loading spectra info
        # NOTE: We ALWAYS cache dataframes at runtime (regardless of cache_dataframes flag)
        # to avoid re-reading files for every sample. The flag only controls initial caching.
        self.df_cache = {}
        
        # Use shared spectra info if provided (for train/val/test split efficiency)
        if shared_spectra_info is not None:
            file_indices, row_indices, self.parquet_files = shared_spectra_info
            self.file_indices = file_indices
            self.row_indices = row_indices
            total_spectra = len(file_indices)
            print(f"Using shared spectra info: {total_spectra} total spectra")
        else:
            # Load file list
            if parquet_files is not None:
                self.parquet_files = parquet_files
            elif metadata_file and os.path.exists(metadata_file):
                self.parquet_files = self._load_from_metadata(metadata_file)
            else:
                self.parquet_files = self._discover_parquet_files(data_dir)
            
            # Limit number of files if specified
            if max_files:
                self.parquet_files = self.parquet_files[:max_files]
            
            print(f"Found {len(self.parquet_files)} Parquet files")
            
            # Load all spectra information using optimized method
            print(f"Loading spectra info (reading only metadata, not full files)...")
            self.file_indices, self.row_indices = self._load_spectra_info()
            
            total_spectra = len(self.file_indices)
            print(f"Total spectra: {total_spectra}")
        
        # Split dataset
        print(f"Splitting dataset for {split} split...")
        self.indices = self._split_dataset(val_split, test_split)
        
        print(f"{split.capitalize()} split: {len(self.indices)} spectra")
    
    def _load_from_metadata(self, metadata_file: str) -> List[str]:
        """Load file list from metadata JSON."""
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        return [f['absolute_path'] for f in metadata['files']]
    
    def _discover_parquet_files(self, data_dir: str) -> List[str]:
        """Discover all Parquet files in directory."""
        parquet_files = []
        
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.parquet'):
                    parquet_files.append(os.path.join(root, file))
        
        return sorted(parquet_files)
    
    def _load_spectra_info(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load information about all spectra from Parquet files.
        
        HIGHLY OPTIMIZED: Uses PyArrow to read only metadata (not data).
        Returns compact numpy arrays instead of list of dicts.
        
        Returns:
            Tuple of (file_indices, row_indices) as numpy int32 arrays
        """
        import time
        
        # Pre-allocate lists (will convert to numpy at end)
        file_indices_list = []
        row_indices_list = []
        
        print(f"  Scanning {len(self.parquet_files)} parquet files (metadata only)...")
        total_start = time.time()
        
        for file_idx, parquet_path in enumerate(self.parquet_files):
            try:
                start = time.time()
                
                # HIGHLY OPTIMIZED: Use PyArrow to read ONLY metadata (not data)
                # This is 100-1000x faster than pd.read_parquet()!
                pq_file = pq.ParquetFile(parquet_path)
                num_spectra = pq_file.metadata.num_rows
                
                elapsed = time.time() - start
                print(f"  [{file_idx+1}/{len(self.parquet_files)}] {os.path.basename(parquet_path)}: {num_spectra} spectra ({elapsed:.3f}s)")
                
                # Create arrays for this file (much faster than list comprehension)
                file_indices_list.append(np.full(num_spectra, file_idx, dtype=np.int32))
                row_indices_list.append(np.arange(num_spectra, dtype=np.int32))
                
                # Optionally pre-cache DataFrame if requested
                if self.cache_dataframes:
                    df = pd.read_parquet(parquet_path)
                    self.df_cache[file_idx] = df
                    print(f"    Pre-cached DataFrame in memory")
                
            except Exception as e:
                print(f"Warning: Could not load metadata from {parquet_path}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Concatenate all arrays (single operation, very fast)
        file_indices = np.concatenate(file_indices_list) if file_indices_list else np.array([], dtype=np.int32)
        row_indices = np.concatenate(row_indices_list) if row_indices_list else np.array([], dtype=np.int32)
        
        total_time = time.time() - total_start
        print(f"  Loaded metadata for {len(file_indices)} total spectra in {total_time:.2f}s")
        print(f"  Memory usage: ~{(file_indices.nbytes + row_indices.nbytes) / 1024 / 1024:.1f} MB")
        
        return file_indices, row_indices
    
    def _split_dataset(self, val_split: float, test_split: float) -> np.ndarray:
        """
        Split dataset into train/val/test.
        
        Returns:
            Numpy array of indices for this split
        """
        total = len(self.file_indices)
        
        # Create indices
        indices = np.arange(total, dtype=np.int32)
        np.random.seed(42)  # For reproducibility
        np.random.shuffle(indices)
        
        # Calculate split points
        test_size = int(total * test_split)
        val_size = int(total * val_split)
        train_size = total - val_size - test_size
        
        if self.split == 'train':
            return indices[:train_size]
        elif self.split == 'val':
            return indices[train_size:train_size + val_size]
        elif self.split == 'test':
            return indices[train_size + val_size:]
        else:
            return indices
    
    def _load_spectrum(self, idx: int) -> Dict:
        """
        Load a single spectrum from Parquet file.
        
        Args:
            idx: Index in the dataset
            
        Returns:
            Dictionary with spectrum data
        """
        # Get spectrum info from numpy arrays (very fast!)
        global_idx = self.indices[idx]
        file_idx = int(self.file_indices[global_idx])
        row_idx = int(self.row_indices[global_idx])
        file_path = self.parquet_files[file_idx]
        
        # Load DataFrame (from cache or disk)
        # CRITICAL: Always cache dataframes even if cache_dataframes=False in __init__
        # Otherwise we re-read the file for EVERY sample (extremely slow!)
        if file_idx in self.df_cache:
            df = self.df_cache[file_idx]
        else:
            df = pd.read_parquet(file_path)
            # Cache it to avoid re-reading for next sample from same file
            self.df_cache[file_idx] = df
        
        # Get row data
        row = df.iloc[row_idx]
        
        # Extract data (adjust column names based on your actual Parquet structure)
        # Common column names: sequence, precursor_mz, charge, mz, intensity
        try:
            sequence = row['modified_sequence']
            if isinstance(sequence, bytes):
                sequence = sequence.decode('utf-8')
            sequence = str(sequence)
            
            precursor_mz = float(row.get('precursor_mz', row.get('prec_mz', 500.0)))
            charge = int(row.get('charge', row.get('precursor_charge', 2)))
            
            # Handle m/z and intensity (might be lists, arrays, or strings)
            mz = row.get('mz', row.get('mz_array', []))
            intensity = row.get('intensity', row.get('intensity_array', row.get('intensities', [])))
            
            # Convert to numpy arrays if needed
            if isinstance(mz, str):
                # Use json.loads instead of eval - 10-100x faster!
                mz = np.array(json.loads(mz), dtype=np.float32)
            elif isinstance(mz, list):
                mz = np.array(mz, dtype=np.float32)
            elif not isinstance(mz, np.ndarray):
                mz = np.asarray(mz, dtype=np.float32)
            else:
                mz = mz.astype(np.float32, copy=False)
            
            if isinstance(intensity, str):
                # Use json.loads instead of eval - 10-100x faster!
                intensity = np.array(json.loads(intensity), dtype=np.float32)
            elif isinstance(intensity, list):
                intensity = np.array(intensity, dtype=np.float32)
            elif not isinstance(intensity, np.ndarray):
                intensity = np.asarray(intensity, dtype=np.float32)
            else:
                intensity = intensity.astype(np.float32, copy=False)
            
        except Exception as e:
            # Fallback: return dummy data
            print(f"Warning: Could not load spectrum {row_idx} from {file_path}: {e}")
            return {
                'sequence': 'PEPTIDE',
                'precursor_mz': 500.0,
                'charge': 2,
                'mz': np.array([100.0, 200.0, 300.0]),
                'intensity': np.array([100.0, 200.0, 150.0])
            }
        
        return {
            'sequence': sequence,
            'precursor_mz': precursor_mz,
            'charge': charge,
            'mz': mz,
            'intensity': intensity
        }
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing processed spectrum data
        """
        data = self._load_spectrum(idx)
        
        # Extract data
        sequence = data['sequence']
        precursor_mz = data['precursor_mz']
        charge = data['charge']
        mz = data['mz']
        intensity = data['intensity']
        
        # Tokenize sequence
        tokens = self.tokenizer.encode(sequence, max_length=self.max_length, padding=True)
        tokens = torch.LongTensor(tokens)
        
        # Create attention mask
        sequence_mask = tokens != self.tokenizer.pad_idx
        
        # Prepare metadata (no normalization applied)
        precursor_mz_val, charge_val = self.preprocessor.prepare_metadata(precursor_mz, charge)
        
        # Prepare targets
        target_mz, target_intensity, target_mask = self.preprocessor.prepare_target(mz, intensity)
        
        # Store original max intensity for denormalization
        max_intensity = np.max(intensity) if len(intensity) > 0 else 1.0
        
        return {
            'sequence_tokens': tokens,
            'sequence_mask': sequence_mask,
            'precursor_mz': torch.tensor([precursor_mz_val], dtype=torch.float32).squeeze(),
            'charge': torch.tensor([charge_val], dtype=torch.long).squeeze(),
            'target_mz': torch.from_numpy(target_mz),
            'target_intensity': torch.from_numpy(target_intensity),
            'target_mask': torch.from_numpy(target_mask),
            'max_intensity': torch.tensor([max_intensity], dtype=torch.float32).squeeze(),
        }


def create_parquet_dataloaders(
    data_dir: str,
    metadata_file: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    max_mz: float = 2000.0,
    top_k: int = 200,
    num_predictions: int = 100,
    max_length: int = 50,
    tokenizer: Optional[AminoAcidTokenizer] = None,
    preprocessor: Optional[SpectrumPreprocessor] = None,
    **kwargs
):
    """
    Create train/val/test dataloaders from Parquet data.
    
    OPTIMIZED: Loads spectra info only ONCE and shares it across train/val/test datasets.
    This avoids reading file metadata 3 times, saving significant time and memory.
    
    Args:
        data_dir: Directory containing Parquet files
        metadata_file: Optional metadata JSON file
        batch_size: Batch size
        num_workers: Number of data loading workers
        max_mz: Maximum m/z value for normalization
        top_k: Number of top peaks to extract from spectrum
        num_predictions: Number of predictions the model will make
        max_length: Maximum sequence length
        tokenizer: Optional shared tokenizer
        preprocessor: Optional shared preprocessor
        **kwargs: Additional arguments for ParquetMSDataset (e.g., cache_dataframes, max_files)
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from torch.utils.data import DataLoader
    import time
    
    print("\n" + "="*80)
    print("CREATING DATALOADERS (OPTIMIZED)")
    print("="*80)
    
    start_time = time.time()
    
    # Step 1: Load file list once
    print("\nStep 1: Discovering Parquet files...")
    temp_dataset = ParquetMSDataset(
        data_dir=data_dir,
        metadata_file=metadata_file,
        max_mz=max_mz,
        top_k=top_k,
        num_predictions=num_predictions,
        max_length=max_length,
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        **kwargs
    )
    
    # Extract the loaded info
    parquet_files = temp_dataset.parquet_files
    file_indices = temp_dataset.file_indices
    row_indices = temp_dataset.row_indices
    
    # Create shared spectra info tuple
    shared_spectra_info = (file_indices, row_indices, parquet_files)
    
    print(f"\nStep 2: Creating train/val/test splits (sharing spectra info)...")
    
    # Get splits info from kwargs
    val_split = kwargs.pop('val_split', 0.1)
    test_split = kwargs.pop('test_split', 0.1)
    cache_dataframes = kwargs.pop('cache_dataframes', False)
    
    # Create datasets with shared spectra info (no re-reading files!)
    train_dataset = ParquetMSDataset(
        split='train',
        val_split=val_split,
        test_split=test_split,
        max_mz=max_mz, 
        top_k=top_k, 
        num_predictions=num_predictions,
        max_length=max_length,
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        cache_dataframes=cache_dataframes,
        shared_spectra_info=shared_spectra_info,
        **kwargs
    )
    
    val_dataset = ParquetMSDataset(
        split='val',
        val_split=val_split,
        test_split=test_split,
        max_mz=max_mz, 
        top_k=top_k, 
        num_predictions=num_predictions,
        max_length=max_length,
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        cache_dataframes=cache_dataframes,
        shared_spectra_info=shared_spectra_info,
        **kwargs
    )
    
    test_dataset = ParquetMSDataset(
        split='test',
        val_split=val_split,
        test_split=test_split,
        max_mz=max_mz, 
        top_k=top_k, 
        num_predictions=num_predictions,
        max_length=max_length,
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        cache_dataframes=cache_dataframes,
        shared_spectra_info=shared_spectra_info,
        **kwargs
    )
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")
    
    # Step 3: Create dataloaders
    print(f"\nStep 3: Creating DataLoaders...")
    
    # Adjust num_workers based on whether persistent_workers should be used
    use_persistent = num_workers > 0
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True, 
        persistent_workers=use_persistent
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=use_persistent
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=use_persistent
    )
    
    elapsed_time = time.time() - start_time
    print(f"\n✓ DataLoaders created in {elapsed_time:.2f}s")
    print("="*80 + "\n")
    
    return train_loader, val_loader, test_loader

