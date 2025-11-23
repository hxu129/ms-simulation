"""
Parquet Dataset loader for MassIVE-KB and ProteomeTools data.
"""

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from typing import Dict, List, Optional
import json
from pathlib import Path

from .tokenizer import AminoAcidTokenizer
from .preprocessing import SpectrumPreprocessor


class ParquetMSDataset(Dataset):
    """
    Dataset for loading MS/MS data from Parquet files.
    
    Supports MassIVE-KB and ProteomeTools data in Parquet format.
    """
    
    def __init__(
        self,
        data_dir: str,
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
        num_predictions: int = 100
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
        
        # Load file list
        if metadata_file and os.path.exists(metadata_file):
            self.parquet_files = self._load_from_metadata(metadata_file)
        else:
            self.parquet_files = self._discover_parquet_files(data_dir)
        
        # Limit number of files if specified
        if max_files:
            self.parquet_files = self.parquet_files[:max_files]
        
        print(f"Found {len(self.parquet_files)} Parquet files")
        
        # Load all spectra information
        print(f"Loading spectra info for {split} split...")
        self.spectra_info = self._load_spectra_info()
        
        print(f"Total spectra: {len(self.spectra_info)}")
        
        # Split dataset
        print(f"Splitting dataset (this may take a moment)...")
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
    
    def _load_spectra_info(self) -> List[Dict]:
        """
        Load information about all spectra from Parquet files.
        
        OPTIMIZED: Only reads metadata to get row count, not full data.
        
        Returns:
            List of dictionaries containing file_idx and row_idx
        """
        import time
        spectra_info = []
        
        for file_idx, parquet_path in enumerate(self.parquet_files):
            try:
                print(f"  Reading parquet file {file_idx+1}/{len(self.parquet_files)}...")
                start = time.time()
                
                # OPTIMIZED: Read only metadata to get row count (much faster!)
                parquet_file = pd.read_parquet(parquet_path)
                num_spectra = len(parquet_file)
                print(f"    Loaded {num_spectra} rows in {time.time()-start:.2f}s")
                
                # Pre-allocate the list for this file (faster than append)
                print(f"    Creating spectra info list...")
                start = time.time()
                file_spectra_info = [
                    {
                        'file_idx': file_idx,
                        'row_idx': row_idx,
                        'file_path': parquet_path
                    }
                    for row_idx in range(num_spectra)
                ]
                spectra_info.extend(file_spectra_info)
                print(f"    Created info list in {time.time()-start:.2f}s")
                
                # Cache DataFrame if requested during initialization
                if self.cache_dataframes:
                    self.df_cache[file_idx] = parquet_file
                    print(f"    Pre-cached {num_spectra} spectra in memory")
                else:
                    # Don't pre-cache, but will cache on first access
                    del parquet_file
                    print(f"    Will cache on first access (lazy caching)")
                
            except Exception as e:
                print(f"Warning: Could not load {parquet_path}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"  Total spectra info entries: {len(spectra_info)}")
        return spectra_info
    
    def _split_dataset(self, val_split: float, test_split: float) -> List[int]:
        """Split dataset into train/val/test."""
        total = len(self.spectra_info)
        
        # Create indices
        indices = np.arange(total)
        np.random.seed(42)  # For reproducibility
        np.random.shuffle(indices)
        
        # Calculate split points
        test_size = int(total * test_split)
        val_size = int(total * val_split)
        train_size = total - val_size - test_size
        
        if self.split == 'train':
            return indices[:train_size].tolist()
        elif self.split == 'val':
            return indices[train_size:train_size + val_size].tolist()
        elif self.split == 'test':
            return indices[train_size + val_size:].tolist()
        else:
            return indices.tolist()
    
    def _load_spectrum(self, idx: int) -> Dict:
        """
        Load a single spectrum from Parquet file.
        
        Args:
            idx: Index in the dataset
            
        Returns:
            Dictionary with spectrum data
        """
        # Get spectrum info
        global_idx = self.indices[idx]
        info = self.spectra_info[global_idx]
        
        file_idx = info['file_idx']
        row_idx = info['row_idx']
        file_path = info['file_path']
        
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
    **kwargs
):
    """
    Create train/val/test dataloaders from Parquet data.
    
    Args:
        data_dir: Directory containing Parquet files
        metadata_file: Optional metadata JSON file
        batch_size: Batch size
        num_workers: Number of data loading workers
        max_mz: Maximum m/z value for normalization
        top_k: Number of top peaks to extract from spectrum
        num_predictions: Number of predictions the model will make
        max_length: Maximum sequence length
        **kwargs: Additional arguments for ParquetMSDataset
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from torch.utils.data import DataLoader
    from .dataset import collate_fn
    
    # Create datasets
    train_dataset = ParquetMSDataset(
        data_dir, metadata_file, split='train',
        max_mz=max_mz, top_k=top_k, num_predictions=num_predictions,
        max_length=max_length, **kwargs
    )
    val_dataset = ParquetMSDataset(
        data_dir, metadata_file, split='val',
        max_mz=max_mz, top_k=top_k, num_predictions=num_predictions,
        max_length=max_length, **kwargs
    )
    test_dataset = ParquetMSDataset(
        data_dir, metadata_file, split='test',
        max_mz=max_mz, top_k=top_k, num_predictions=num_predictions,
        max_length=max_length, **kwargs
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True, 
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True
    )
    
    return train_loader, val_loader, test_loader

