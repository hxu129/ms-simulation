"""
Cosine similarity loss for global distribution consistency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


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
    ) -> torch.Tensor:
        """
        Create binned spectrum representation.
        
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
        pred_mz: torch.Tensor,
        pred_intensity: torch.Tensor,
        target_mz: torch.Tensor,
        target_intensity: torch.Tensor,
        target_mask: torch.Tensor,
        matched_indices: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Compute cosine similarity loss.
        
        Args:
            pred_mz: Predicted m/z values, shape (batch_size, num_predictions)
            pred_intensity: Predicted intensities, shape (batch_size, num_predictions)
            target_mz: Target m/z values, shape (batch_size, num_targets)
            target_intensity: Target intensities, shape (batch_size, num_targets)
            target_mask: Mask for real targets, shape (batch_size, num_targets)
            matched_indices: Optional tuple of (batch_idx, pred_idx, target_idx) from Hungarian matching.
                           If provided, only matched predictions are used for cosine similarity.
            
        Returns:
            Cosine similarity loss (1 - cosine_similarity)
        """
        
        batch_size = pred_mz.size(0)
        device = pred_mz.device
        
        # --- 优化部分开始 ---
        
        # 1. 处理 Predictions
        if matched_indices is not None:
            # 方案 A: 只使用匹配上的 Predictions (极速版)
            batch_idx, pred_idx, target_idx = matched_indices
            
            # 直接提取匹配的值 (Flattened)
            flat_mz = pred_mz[batch_idx, pred_idx]
            flat_intensity = pred_intensity[batch_idx, pred_idx]
            
            # 计算 Bin Index
            mz_denorm = flat_mz * self.max_mz
            bin_idx = (mz_denorm / self.bin_size).long().clamp(0, self.num_bins - 1)
            
            # 核心 Trick: 计算全局 Flattened Index
            # Index = 当前样本号 * 总Bin数 + 当前Bin号
            # 这样就把 (Batch, Bin) 映射到了 1D 空间
            flat_indices = batch_idx * self.num_bins + bin_idx
            
            # 初始化打平的 Binned Spectrum
            pred_binned_flat = torch.zeros(batch_size * self.num_bins, device=device, dtype=pred_intensity.dtype)
            
            # 一次性 Scatter 所有数据 (无需 Loop)
            pred_binned_flat.scatter_add_(0, flat_indices, flat_intensity)
            
            # Reshape 回 (Batch, Num_Bins)
            pred_binned = pred_binned_flat.view(batch_size, self.num_bins)
            
            # Normalization
            if self.normalize_bins:
                norm = pred_binned.norm(p=2, dim=1, keepdim=True)
                pred_binned = pred_binned / (norm + 1e-8)
                
        else:
            # 方案 B: 使用原始逻辑 (也可以优化，但保持原样也没问题)
            pred_binned = self.create_binned_spectrum(pred_mz, pred_intensity)

        # --- 优化部分结束 ---

        # 2. 处理 Targets (保持不变)
        target_intensity_masked = target_intensity * target_mask
        target_binned = self.create_binned_spectrum(target_mz, target_intensity_masked)
        
        # 3. 计算 Loss
        cosine_sim = F.cosine_similarity(pred_binned, target_binned, dim=1)
        
        if isinstance(self, CosineSimilarityLoss):
            loss = 1.0 - cosine_sim.mean()
        else: # SpectralAngleLoss
            cosine_sim = torch.clamp(cosine_sim, -1.0, 1.0)
            angle = torch.acos(cosine_sim)
            loss = (angle / torch.pi).mean()
            
        # return self.weight * loss
        return loss


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
    ) -> torch.Tensor:
        """Create binned spectrum (same as CosineSimilarityLoss)."""
        batch_size = mz.size(0)
        device = mz.device
        
        mz_denorm = mz * self.max_mz
        bin_indices = (mz_denorm / self.bin_size).long()
        bin_indices = torch.clamp(bin_indices, 0, self.num_bins - 1)
        
        binned = torch.zeros(batch_size, self.num_bins, device=device)
        for b in range(batch_size):
            binned[b].scatter_add_(0, bin_indices[b], intensity[b])
        
        return binned
    
    def forward(
        self,
        pred_mz: torch.Tensor,
        pred_intensity: torch.Tensor,
        target_mz: torch.Tensor,
        target_intensity: torch.Tensor,
        target_mask: torch.Tensor,
        matched_indices: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Compute spectral angle loss.
        
        Args:
            matched_indices: Optional tuple of (batch_idx, pred_idx, target_idx) from Hungarian matching.
                           If provided, only matched predictions are used.
        
        Returns:
            Spectral angle in radians (normalized by π)
        """
        # Filter predictions to only matched ones if indices provided
        if matched_indices is not None:
            batch_idx, pred_idx, target_idx = matched_indices
            batch_size = pred_mz.size(0)
            device = pred_mz.device
            
            # Reconstruct matched predictions into (batch_size, max_matches) format
            matches_per_sample = torch.bincount(batch_idx, minlength=batch_size)
            max_matches = matches_per_sample.max().item() if len(batch_idx) > 0 else 0
            
            if max_matches > 0:
                # Create padded tensors for matched predictions
                matched_pred_mz_padded = torch.zeros(batch_size, max_matches, device=device, dtype=pred_mz.dtype)
                matched_pred_intensity_padded = torch.zeros(batch_size, max_matches, device=device, dtype=pred_intensity.dtype)
                
                # Fill in the matched predictions using vectorized indexing
                # Compute position within each batch for each match
                positions = torch.zeros_like(batch_idx)
                for b in torch.unique(batch_idx):
                    mask = batch_idx == b
                    positions[mask] = torch.arange(mask.sum(), device=device)
                
                # Extract matched values and assign using advanced indexing
                matched_pred_mz_padded[batch_idx, positions] = pred_mz[batch_idx, pred_idx]
                matched_pred_intensity_padded[batch_idx, positions] = pred_intensity[batch_idx, pred_idx]
                
                # Use create_binned_spectrum for matched predictions
                pred_binned = self.create_binned_spectrum(matched_pred_mz_padded, matched_pred_intensity_padded)
            else:
                # No matches, create empty binned spectrum
                pred_binned = torch.zeros(batch_size, self.num_bins, device=device)
        else:
            # Use all predictions (original behavior)
            pred_binned = self.create_binned_spectrum(pred_mz, pred_intensity)
        
        target_intensity_masked = target_intensity * target_mask
        target_binned = self.create_binned_spectrum(target_mz, target_intensity_masked)
        
        # Compute cosine similarity
        cosine_sim = F.cosine_similarity(pred_binned, target_binned, dim=1)
        cosine_sim = torch.clamp(cosine_sim, -1.0, 1.0)
        
        # Compute angle in radians
        angle = torch.acos(cosine_sim)
        
        # Normalize by pi and apply weight
        loss = (angle / torch.pi).mean()
        
        return self.weight * loss

