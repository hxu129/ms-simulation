"""
Cosine similarity loss for global distribution consistency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CosineSimilarityLoss(nn.Module):
    """
    Cosine similarity loss for comparing predicted and target spectrum distributions.
    
    This loss enforces global distribution consistency between the predicted spectrum
    and the target spectrum, complementing the Hungarian matching loss which focuses
    on individual peak matching.
    """
    
    def __init__(
        self,
        weight: float = 1.0,
        bin_size: float = 1.0,
        max_mz: float = 2000.0,
        normalize_bins: bool = True
    ):
        """
        Initialize cosine similarity loss.
        
        Args:
            weight: Tunable scaler for this loss component
            bin_size: Size of m/z bins in Da
            max_mz: Maximum m/z value
            normalize_bins: Whether to normalize binned spectra
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
        confidence: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Create binned spectrum representation.
        
        Args:
            mz: m/z values (normalized to [0, 1]), shape (batch_size, num_peaks)
            intensity: Intensity values, shape (batch_size, num_peaks)
            confidence: Confidence scores (optional), shape (batch_size, num_peaks)
            
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
        
        # Weight intensity by confidence if provided
        if confidence is not None:
            weighted_intensity = intensity * confidence
        else:
            weighted_intensity = intensity
        
        # Create binned spectrum
        binned = torch.zeros(batch_size, self.num_bins, device=device)
        
        # Scatter add intensities to bins
        for b in range(batch_size):
            binned[b].scatter_add_(0, bin_indices[b], weighted_intensity[b])
        
        # Normalize if requested
        if self.normalize_bins:
            norm = binned.norm(p=2, dim=1, keepdim=True)
            binned = binned / (norm + 1e-8)
        
        return binned
    
    def forward(
        self,
        pred_mz: torch.Tensor,
        pred_intensity: torch.Tensor,
        pred_confidence: torch.Tensor,
        target_mz: torch.Tensor,
        target_intensity: torch.Tensor,
        target_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cosine similarity loss.
        
        Args:
            pred_mz: Predicted m/z values, shape (batch_size, num_predictions)
            pred_intensity: Predicted intensities, shape (batch_size, num_predictions)
            pred_confidence: Predicted confidences, shape (batch_size, num_predictions)
            target_mz: Target m/z values, shape (batch_size, num_targets)
            target_intensity: Target intensities, shape (batch_size, num_targets)
            target_mask: Mask for real targets, shape (batch_size, num_targets)
            
        Returns:
            Cosine similarity loss (1 - cosine_similarity)
        """
        # Create binned spectra
        pred_binned = self.create_binned_spectrum(pred_mz, pred_intensity, pred_confidence)
        
        # For targets, use mask to zero out padding
        target_intensity_masked = target_intensity * target_mask
        target_binned = self.create_binned_spectrum(target_mz, target_intensity_masked)
        
        # Compute cosine similarity
        cosine_sim = F.cosine_similarity(pred_binned, target_binned, dim=1)
        
        # Loss is 1 - cosine_similarity (minimize distance)
        loss = 1.0 - cosine_sim.mean()
        
        return self.weight * loss


class SpectralAngleLoss(nn.Module):
    """
    Spectral angle loss (alternative to cosine similarity).
    
    Computes the angle between predicted and target spectra in high-dimensional space.
    """
    
    def __init__(
        self,
        weight: float = 1.0,
        bin_size: float = 1.0,
        max_mz: float = 2000.0
    ):
        """
        Initialize spectral angle loss.
        
        Args:
            weight: Tunable scaler for this loss component
            bin_size: Size of m/z bins in Da
            max_mz: Maximum m/z value
        """
        super().__init__()
        self.weight = weight
        self.bin_size = bin_size
        self.max_mz = max_mz
        self.num_bins = int(max_mz / bin_size)
    
    def create_binned_spectrum(
        self,
        mz: torch.Tensor,
        intensity: torch.Tensor,
        confidence: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Create binned spectrum (same as CosineSimilarityLoss)."""
        batch_size = mz.size(0)
        device = mz.device
        
        mz_denorm = mz * self.max_mz
        bin_indices = (mz_denorm / self.bin_size).long()
        bin_indices = torch.clamp(bin_indices, 0, self.num_bins - 1)
        
        if confidence is not None:
            weighted_intensity = intensity * confidence
        else:
            weighted_intensity = intensity
        
        binned = torch.zeros(batch_size, self.num_bins, device=device)
        for b in range(batch_size):
            binned[b].scatter_add_(0, bin_indices[b], weighted_intensity[b])
        
        return binned
    
    def forward(
        self,
        pred_mz: torch.Tensor,
        pred_intensity: torch.Tensor,
        pred_confidence: torch.Tensor,
        target_mz: torch.Tensor,
        target_intensity: torch.Tensor,
        target_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute spectral angle loss.
        
        Returns:
            Spectral angle in radians (normalized by π)
        """
        pred_binned = self.create_binned_spectrum(pred_mz, pred_intensity, pred_confidence)
        target_intensity_masked = target_intensity * target_mask
        target_binned = self.create_binned_spectrum(target_mz, target_intensity_masked)
        
        # Compute cosine similarity
        cosine_sim = F.cosine_similarity(pred_binned, target_binned, dim=1)
        cosine_sim = torch.clamp(cosine_sim, -1.0, 1.0)
        
        # Compute angle in radians
        angle = torch.acos(cosine_sim)
        
        # Normalize by π and apply weight
        loss = (angle / torch.pi).mean()
        
        return self.weight * loss

