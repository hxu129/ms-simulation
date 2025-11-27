"""
Cosine similarity loss for binned spectrum prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinnedCosineLoss(nn.Module):
    """
    Cosine similarity loss for binned spectrum prediction.
    
    This loss compares predicted binned spectrum vectors directly with
    ground truth binned spectra created from (mz, intensity) pairs.
    """
    
    def __init__(
        self,
        weight: float = 1.0,
        bin_size: float = 1.0,
        max_mz: float = 1500.0,
        normalize_bins: bool = True
    ):
        """
        Initialize binned cosine loss.
        
        Args:
            weight: Loss weight (multiplier)
            bin_size: Size of m/z bins in Da
            max_mz: Maximum m/z value
            normalize_bins: Whether to L2 normalize binned spectra
        """
        super().__init__()
        self.weight = weight
        self.bin_size = bin_size
        self.max_mz = max_mz
        self.normalize_bins = normalize_bins
        self.num_bins = int(max_mz / bin_size)
    
    def create_binned_spectrum(
        self,
        mz: torch.Tensor,
        intensity: torch.Tensor,
    ) -> torch.Tensor:
        """
        Create binned spectrum representation from m/z and intensity values.
        
        Args:
            mz: m/z values (normalized to [0, 1]), shape (batch_size, num_peaks)
            intensity: Intensity values, shape (batch_size, num_peaks)
            
        Returns:
            Binned spectrum, shape (batch_size, num_bins)
        """
        batch_size = mz.size(0)
        device = mz.device
        
        # Denormalize m/z
        mz_denorm = mz * self.max_mz
        
        # Convert m/z to bin indices
        bin_indices = (mz_denorm / self.bin_size).long()
        bin_indices = torch.clamp(bin_indices, 0, self.num_bins - 1)
        
        # Create binned spectrum
        binned = torch.zeros(batch_size, self.num_bins, device=device, dtype=intensity.dtype)
        
        # Scatter add intensities to bins
        for b in range(batch_size):
            binned[b].scatter_add_(0, bin_indices[b], intensity[b])
        
        # Normalize if requested
        if self.normalize_bins:
            norm = binned.norm(p=2, dim=1, keepdim=True)
            binned = binned / (norm + 1e-8)
        
        return binned
    
    def forward(
        self,
        pred_binned: torch.Tensor,
        target_mz: torch.Tensor,
        target_intensity: torch.Tensor,
        target_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cosine similarity loss.
        
        Args:
            pred_binned: Predicted binned spectrum, shape (batch_size, num_bins)
            target_mz: Target m/z values, shape (batch_size, num_targets)
            target_intensity: Target intensities, shape (batch_size, num_targets)
            target_mask: Mask for real targets, shape (batch_size, num_targets)
            
        Returns:
            Cosine similarity loss (1 - cosine_similarity)
        """
        # Normalize predicted binned spectrum
        if self.normalize_bins:
            pred_norm = pred_binned.norm(p=2, dim=1, keepdim=True)
            pred_binned_normalized = pred_binned / (pred_norm + 1e-8)
        else:
            pred_binned_normalized = pred_binned
        
        # Create ground truth binned spectrum
        target_intensity_masked = target_intensity * target_mask
        target_binned = self.create_binned_spectrum(target_mz, target_intensity_masked)
        
        # Compute cosine similarity
        cosine_sim = F.cosine_similarity(pred_binned_normalized, target_binned, dim=1)
        
        # Compute loss (1 - cosine similarity)
        loss = 1.0 - cosine_sim.mean()
        
        return self.weight * loss

