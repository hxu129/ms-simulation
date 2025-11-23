"""
HDF5 Dataset loader for MassIVE-KB and ProteomeTools data.
"""

import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List, Optional
import json

from .tokenizer import AminoAcidTokenizer
from .preprocessing import SpectrumPreprocessor


class HDF5MSDataset(Dataset):
    """
    Dataset for loading MS/MS data from HDF5 files.
    
    Supports MassIVE-KB and ProteomeTools data formats.
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
        cache_in_memory: bool = False,
        max_mz: float = 2000.0,
        top_k: int = 200,
        num_predictions: int = 100
    ):
        """
        Initialize HDF5 dataset.
        
        Args:
            data_dir: Directory containing HDF5 files
            metadata_file: Optional JSON metadata file listing all HDF5 files
            tokenizer: Amino acid tokenizer
            preprocessor: Spectrum preprocessor
            max_length: Maximum sequence length
            split: Dataset split ('train', 'val', 'test')
            val_split: Fraction of data for validation
            test_split: Fraction of data for testing
            cache_in_memory: Whether to cache all data in memory
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
        self.cache_in_memory = cache_in_memory
        
        # Load file list
        if metadata_file and os.path.exists(metadata_file):
            self.hdf5_files = self._load_from_metadata(metadata_file)
        else:
            self.hdf5_files = self._discover_hdf5_files(data_dir)
        
        print(f"Found {len(self.hdf5_files)} HDF5 files")
        
        # Load all spectra information
        self.spectra_info = self._load_spectra_info()
        
        print(f"Total spectra: {len(self.spectra_info)}")
        
        # Split dataset
        self.indices = self._split_dataset(val_split, test_split)
        
        print(f"{split.capitalize()} split: {len(self.indices)} spectra")
        
        # Cache data if requested
        self.cache = {}
        if cache_in_memory:
            print("Caching data in memory...")
            for idx in range(len(self.indices)):
                self.cache[idx] = self._load_spectrum(idx)
            print("Caching complete")
    
    def _load_from_metadata(self, metadata_file: str) -> List[str]:
        """Load file list from metadata JSON."""
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        return [f['absolute_path'] for f in metadata['files']]
    
    def _discover_hdf5_files(self, data_dir: str) -> List[str]:
        """Discover all HDF5 files in directory."""
        hdf5_files = []
        
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.hdf5'):
                    hdf5_files.append(os.path.join(root, file))
        
        return sorted(hdf5_files)
    
    def _load_spectra_info(self) -> List[Dict]:
        """
        Load information about all spectra from HDF5 files.
        
        Returns:
            List of dictionaries containing file_idx and spectrum_idx
        """
        spectra_info = []
        
        for file_idx, hdf5_path in enumerate(self.hdf5_files):
            try:
                with h5py.File(hdf5_path, 'r') as f:
                    # Determine the structure of HDF5 file
                    # This needs to be adapted based on actual HDF5 structure
                    
                    # Common structures:
                    # Option 1: Each spectrum is a group
                    # Option 2: Arrays of sequences, mz, intensity, etc.
                    
                    # Try to detect structure
                    if 'sequences' in f or 'sequence' in f:
                        # Array-based structure
                        num_spectra = len(f.get('sequences', f.get('sequence', [])))
                    elif 'spectra' in f:
                        # Group-based structure
                        num_spectra = len(f['spectra'].keys())
                    else:
                        # Iterate through top-level to count spectra
                        num_spectra = len([k for k in f.keys() if isinstance(f[k], h5py.Group)])
                    
                    for spectrum_idx in range(num_spectra):
                        spectra_info.append({
                            'file_idx': file_idx,
                            'spectrum_idx': spectrum_idx,
                            'file_path': hdf5_path
                        })
            
            except Exception as e:
                print(f"Warning: Could not load {hdf5_path}: {e}")
                continue
        
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
        Load a single spectrum from HDF5 file.
        
        Args:
            idx: Index in the dataset
            
        Returns:
            Dictionary with spectrum data
        """
        # Get spectrum info
        global_idx = self.indices[idx]
        info = self.spectra_info[global_idx]
        
        file_path = info['file_path']
        spectrum_idx = info['spectrum_idx']
        
        # Open HDF5 file and load spectrum
        with h5py.File(file_path, 'r') as f:
            # This needs to be adapted based on actual HDF5 structure
            # Here are some common patterns:
            
            try:
                # Pattern 1: Array-based (all data in arrays)
                if 'sequences' in f or 'sequence' in f:
                    sequence_key = 'sequences' if 'sequences' in f else 'sequence'
                    sequence = f[sequence_key][spectrum_idx]
                    
                    # Decode if bytes
                    if isinstance(sequence, bytes):
                        sequence = sequence.decode('utf-8')
                    elif isinstance(sequence, np.ndarray):
                        sequence = sequence.item().decode('utf-8') if sequence.dtype == 'O' else str(sequence)
                    
                    # Load metadata
                    precursor_mz = float(f['precursor_mz'][spectrum_idx]) if 'precursor_mz' in f else 500.0
                    charge = int(f['charge'][spectrum_idx]) if 'charge' in f else 2
                    
                    # Load spectrum
                    mz = np.array(f['mz'][spectrum_idx]) if 'mz' in f else np.array([])
                    intensity = np.array(f['intensity'][spectrum_idx]) if 'intensity' in f else np.array([])
                
                # Pattern 2: Group-based (each spectrum is a group)
                elif 'spectra' in f:
                    spectrum_group = f['spectra'][str(spectrum_idx)]
                    
                    sequence = spectrum_group['sequence'][()]
                    if isinstance(sequence, bytes):
                        sequence = sequence.decode('utf-8')
                    
                    precursor_mz = float(spectrum_group['precursor_mz'][()])
                    charge = int(spectrum_group['charge'][()])
                    mz = np.array(spectrum_group['mz'])
                    intensity = np.array(spectrum_group['intensity'])
                
                # Pattern 3: Top-level groups (each group is a spectrum)
                else:
                    keys = [k for k in f.keys() if isinstance(f[k], h5py.Group)]
                    spectrum_key = keys[spectrum_idx]
                    spectrum_group = f[spectrum_key]
                    
                    sequence = spectrum_group['sequence'][()]
                    if isinstance(sequence, bytes):
                        sequence = sequence.decode('utf-8')
                    
                    precursor_mz = float(spectrum_group.get('precursor_mz', [500.0])[()])
                    charge = int(spectrum_group.get('charge', [2])[()])
                    mz = np.array(spectrum_group['mz'])
                    intensity = np.array(spectrum_group['intensity'])
            
            except Exception as e:
                # Fallback: return dummy data
                print(f"Warning: Could not load spectrum {spectrum_idx} from {file_path}: {e}")
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
        # Check cache first
        if idx in self.cache:
            data = self.cache[idx]
        else:
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
        
        # Prepare metadata
        norm_precursor_mz, charge_val = self.preprocessor.prepare_metadata(precursor_mz, charge)
        
        # Prepare targets (already returns float32 numpy arrays)
        target_mz, target_intensity, target_mask = self.preprocessor.prepare_target(mz, intensity)
        
        # Store original max intensity for denormalization
        max_intensity = np.max(intensity) if len(intensity) > 0 else 1.0
        
        # OPTIMIZED: Use torch.from_numpy() for zero-copy conversion (2-3x faster)
        return {
            'sequence_tokens': tokens,
            'sequence_mask': sequence_mask,
            'precursor_mz': torch.tensor([norm_precursor_mz], dtype=torch.float32).squeeze(),
            'charge': torch.tensor([charge_val], dtype=torch.long).squeeze(),
            'target_mz': torch.from_numpy(target_mz),
            'target_intensity': torch.from_numpy(target_intensity),
            'target_mask': torch.from_numpy(target_mask),
            'max_intensity': torch.tensor([max_intensity], dtype=torch.float32).squeeze(),
        }


def create_hdf5_dataloaders(
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
    Create train/val/test dataloaders from HDF5 data.
    
    Args:
        data_dir: Directory containing HDF5 files
        metadata_file: Optional metadata JSON file
        batch_size: Batch size
        num_workers: Number of data loading workers
        max_mz: Maximum m/z value for normalization
        top_k: Number of top peaks to extract from spectrum
        num_predictions: Number of predictions the model will make
        max_length: Maximum sequence length
        **kwargs: Additional arguments for HDF5MSDataset
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from torch.utils.data import DataLoader
    from .dataset import collate_fn
    
    # Create datasets
    train_dataset = HDF5MSDataset(
        data_dir, metadata_file, split='train',
        max_mz=max_mz, top_k=top_k, num_predictions=num_predictions,
        max_length=max_length, **kwargs
    )
    val_dataset = HDF5MSDataset(
        data_dir, metadata_file, split='val',
        max_mz=max_mz, top_k=top_k, num_predictions=num_predictions,
        max_length=max_length, **kwargs
    )
    test_dataset = HDF5MSDataset(
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
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

