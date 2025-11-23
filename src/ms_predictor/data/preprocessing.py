"""
Preprocessing functions for mass spectrometry data.
"""

import numpy as np
from typing import Tuple, List, Optional


class SpectrumPreprocessor:
    """
    Preprocessing utilities for MS/MS spectra.
    
    Handles normalization, peak extraction, and target construction.
    """
    
    def __init__(
        self,
        max_mz: float = 2000.0,
        top_k: int = 200,
        num_predictions: int = 100
    ):
        """
        Initialize the preprocessor.
        
        Args:
            max_mz: Maximum m/z value for normalization
            top_k: Number of top peaks to extract from spectrum
            num_predictions: Number of predictions the model will make (N)
        """
        self.max_mz = max_mz
        self.top_k = top_k
        self.num_predictions = num_predictions
    
    def extract_top_k_peaks(
        self,
        mz: np.ndarray,
        intensity: np.ndarray,
        k: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract top-K peaks by intensity.
        
        Args:
            mz: Array of m/z values
            intensity: Array of intensity values
            k: Number of peaks to keep (defaults to self.top_k)
            
        Returns:
            Tuple of (mz_topk, intensity_topk)
        """
        if k is None:
            k = self.top_k
        
        if len(intensity) == 0:
            return np.array([]), np.array([])
        
        # Get indices of top-K intensities
        if len(intensity) <= k:
            # If we have fewer peaks than k, return all
            return mz, intensity
        
        # Get top-k indices
        top_k_indices = np.argpartition(intensity, -k)[-k:]
        top_k_indices = top_k_indices[np.argsort(intensity[top_k_indices])[::-1]]
        
        return mz[top_k_indices], intensity[top_k_indices]
    
    def normalize_mz(self, mz: np.ndarray) -> np.ndarray:
        """
        Normalize m/z values to [0, 1] range.
        
        Args:
            mz: Array of m/z values
            
        Returns:
            Normalized m/z values in [0, 1] range
        """
        return mz / self.max_mz
    
    def denormalize_mz(self, mz_normalized: np.ndarray) -> np.ndarray:
        """
        Denormalize m/z values from [0, 1] back to original scale.
        
        Args:
            mz_normalized: m/z values in [0, 1] range
            
        Returns:
            m/z values in original scale (0 to max_mz)
        """
        return mz_normalized * self.max_mz
    
    def normalize_intensity(self, intensity: np.ndarray) -> np.ndarray:
        """
        Normalize intensities to [0, 1] by dividing by max intensity.
        
        Args:
            intensity: Array of intensity values
            
        Returns:
            Normalized intensity values
        """
        max_intensity = np.max(intensity)
        if max_intensity > 0:
            return intensity / max_intensity
        return intensity
    
    def denormalize_intensity(
        self,
        intensity_normalized: np.ndarray,
        max_intensity: float
    ) -> np.ndarray:
        """
        Denormalize intensities from [0, 1] back to original scale.
        
        Args:
            intensity_normalized: Normalized intensity values
            max_intensity: Original maximum intensity
            
        Returns:
            Original scale intensity values
        """
        return intensity_normalized * max_intensity
    
    def prepare_target(
        self,
        mz: np.ndarray,
        intensity: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare target set for training.
        
        Extracts top-K peaks, normalizes them, and creates target arrays.
        Note: K_real (number of real peaks) can be larger than N (num_predictions).
        The Hungarian matching algorithm will find the optimal N matches from K_real targets.
        
        Args:
            mz: Array of m/z values
            intensity: Array of intensity values
            
        Returns:
            Tuple of (target_mz, target_intensity, target_mask)
            - target_mz: Shape (top_k,), normalized m/z values (padded with 0 if k_real < top_k)
            - target_intensity: Shape (top_k,), normalized intensities (padded with 0 if k_real < top_k)
            - target_mask: Shape (top_k,), 1 for real peaks, 0 for padding
            
            Note: k_real can be anywhere from 0 to top_k. The Hungarian matcher
            will handle the case where k_real != num_predictions.
        """
        # Extract top-K peaks
        mz_topk, intensity_topk = self.extract_top_k_peaks(mz, intensity)
        
        # Normalize
        mz_norm = self.normalize_mz(mz_topk)
        intensity_norm = self.normalize_intensity(intensity_topk)
        
        # Create target arrays padded to top_k (not num_predictions!)
        # This allows k_real to be larger than num_predictions
        k_real = len(mz_norm)
        max_targets = self.top_k  # Use top_k as the max size instead of num_predictions
        
        target_mz = np.zeros(max_targets, dtype=np.float32)
        target_intensity = np.zeros(max_targets, dtype=np.float32)
        target_mask = np.zeros(max_targets, dtype=np.float32)
        
        # Fill in real peaks (no truncation at num_predictions)
        if k_real > 0:
            target_mz[:k_real] = mz_norm[:k_real]
            target_intensity[:k_real] = intensity_norm[:k_real]
            target_mask[:k_real] = 1.0
        
        return target_mz, target_intensity, target_mask
    
    def prepare_metadata(
        self,
        precursor_mz: float,
        charge: int
    ) -> Tuple[float, int]:
        """
        Prepare metadata for model input.
        
        Args:
            precursor_mz: Precursor m/z value
            charge: Charge state
            
        Returns:
            Tuple of (precursor_mz, charge) - no normalization applied
        """
        # Keep precursor m/z as-is (no normalization)
        return precursor_mz, charge


def create_spectrum_bins(
    mz: np.ndarray,
    intensity: np.ndarray,
    bin_size: float = 1.0,
    max_mz: float = 2000.0
) -> np.ndarray:
    """
    Create binned representation of spectrum (alternative representation).
    
    Args:
        mz: Array of m/z values
        intensity: Array of intensity values
        bin_size: Size of each bin in Da
        max_mz: Maximum m/z value
        
    Returns:
        Binned spectrum array
    """
    num_bins = int(max_mz / bin_size)
    binned = np.zeros(num_bins, dtype=np.float32)
    
    for m, i in zip(mz, intensity):
        bin_idx = int(m / bin_size)
        if 0 <= bin_idx < num_bins:
            binned[bin_idx] += i
    
    # Normalize
    max_val = np.max(binned)
    if max_val > 0:
        binned = binned / max_val
    
    return binned

