"""
PyTorch Dataset for mass spectrometry data.
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple
import numpy as np

from .tokenizer import AminoAcidTokenizer
from .preprocessing import SpectrumPreprocessor


class MSDataset(Dataset):
    """
    Dataset for mass spectrometry spectrum prediction.
    
    This is a placeholder implementation that will be filled with actual data loading logic
    when data becomes available.
    """
    
    def __init__(
        self,
        data_path: Optional[str] = None,
        tokenizer: Optional[AminoAcidTokenizer] = None,
        preprocessor: Optional[SpectrumPreprocessor] = None,
        max_length: int = 50,
        split: str = 'train'
    ):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the data file (placeholder for future use)
            tokenizer: Amino acid tokenizer
            preprocessor: Spectrum preprocessor
            max_length: Maximum sequence length
            split: Dataset split ('train', 'val', 'test')
        """
        self.data_path = data_path
        self.tokenizer = tokenizer or AminoAcidTokenizer()
        self.preprocessor = preprocessor or SpectrumPreprocessor()
        self.max_length = max_length
        self.split = split
        
        # Placeholder: Load data here when available
        # For now, create empty lists
        self.sequences = []
        self.precursor_mz = []
        self.charges = []
        self.mz_arrays = []
        self.intensity_arrays = []
        
        if data_path is not None:
            self._load_data()
    
    def _load_data(self):
        """
        Load data from file.
        
        TODO: Implement this method when data becomes available.
        Expected data format:
        - sequences: List of peptide sequences
        - precursor_mz: List of precursor m/z values
        - charges: List of charge states
        - mz_arrays: List of arrays containing fragment m/z values
        - intensity_arrays: List of arrays containing fragment intensities
        """
        # Placeholder for data loading logic
        # Example format (to be replaced):
        # import pandas as pd
        # df = pd.read_csv(self.data_path)
        # self.sequences = df['sequence'].tolist()
        # self.precursor_mz = df['precursor_mz'].tolist()
        # self.charges = df['charge'].tolist()
        # self.mz_arrays = [np.array(eval(mz)) for mz in df['mz']]
        # self.intensity_arrays = [np.array(eval(intensity)) for intensity in df['intensity']]
        pass
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing:
                - sequence_tokens: Encoded peptide sequence (LongTensor)
                - sequence_mask: Attention mask for sequence (BoolTensor)
                - precursor_mz: Normalized precursor m/z (FloatTensor)
                - charge: Charge state (LongTensor)
                - target_mz: Target m/z values (FloatTensor)
                - target_intensity: Target intensities (FloatTensor)
                - target_mask: Mask for real peaks (FloatTensor)
                - max_intensity: Original max intensity for denormalization (FloatTensor)
        """
        # Get raw data
        sequence = self.sequences[idx]
        precursor_mz = self.precursor_mz[idx]
        charge = self.charges[idx]
        mz = self.mz_arrays[idx]
        intensity = self.intensity_arrays[idx]
        
        # Tokenize sequence
        tokens = self.tokenizer.encode(sequence, max_length=self.max_length, padding=True)
        tokens = torch.LongTensor(tokens)
        
        # Create attention mask (1 for real tokens, 0 for padding)
        sequence_mask = tokens != self.tokenizer.pad_idx
        
        # Prepare metadata
        norm_precursor_mz, charge_val = self.preprocessor.prepare_metadata(precursor_mz, charge)
        
        # Prepare targets
        target_mz, target_intensity, target_mask = self.preprocessor.prepare_target(mz, intensity)
        
        # Store original max intensity for denormalization during inference
        max_intensity = np.max(intensity) if len(intensity) > 0 else 1.0
        
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


class DummyMSDataset(Dataset):
    """
    Dummy dataset for testing purposes.
    
    Generates synthetic data to test the model architecture before real data is available.
    """
    
    def __init__(
        self,
        num_samples: int = 100,
        max_length: int = 50,
        num_predictions: int = 100
    ):
        """
        Initialize dummy dataset.
        
        Args:
            num_samples: Number of synthetic samples to generate
            max_length: Maximum sequence length
            num_predictions: Number of predictions (N)
        """
        self.num_samples = num_samples
        self.max_length = max_length
        self.num_predictions = num_predictions
        self.tokenizer = AminoAcidTokenizer()
        self.preprocessor = SpectrumPreprocessor(num_predictions=num_predictions)
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Generate a synthetic sample."""
        np.random.seed(idx)
        
        # Random peptide sequence
        seq_len = np.random.randint(10, self.max_length)
        sequence = ''.join(np.random.choice(list(self.tokenizer.AMINO_ACIDS), size=seq_len))
        
        # Random metadata
        precursor_mz = np.random.uniform(400, 2000)
        charge = np.random.randint(2, 5)
        
        # Random spectrum peaks
        num_peaks = np.random.randint(50, 200)
        mz = np.sort(np.random.uniform(100, 2000, size=num_peaks))
        intensity = np.random.uniform(0, 1000, size=num_peaks)
        
        # Tokenize
        tokens = self.tokenizer.encode(sequence, max_length=self.max_length, padding=True)
        tokens = torch.LongTensor(tokens)
        sequence_mask = tokens != self.tokenizer.pad_idx
        
        # Prepare metadata
        norm_precursor_mz, charge_val = self.preprocessor.prepare_metadata(precursor_mz, charge)
        
        # Prepare targets
        target_mz, target_intensity, target_mask = self.preprocessor.prepare_target(mz, intensity)
        
        max_intensity = np.max(intensity)
        
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

