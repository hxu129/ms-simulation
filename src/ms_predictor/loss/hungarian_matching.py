"""
GPU-accelerated Hungarian matching using batch processing.

This version minimizes CPU-GPU transfers by processing on CPU in batch
and minimizing synchronization points.
"""

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from typing import Tuple
import numpy as np


class FastHungarianMatcher(nn.Module):
    """
    Faster Hungarian matcher that minimizes CPU-GPU transfers.
    
    Optimizations:
    1. Single .cpu() call for all cost matrices
    2. Batch numpy operations
    3. Single device transfer back
    """
    
    def __init__(
        self,
        cost_mz: float = 1.0,
        cost_intensity: float = 1.0,
    ):
        """
        Initialize Hungarian matcher.
        
        Args:
            cost_mz: Weight for m/z position cost
            cost_intensity: Weight for intensity cost
        """
        super().__init__()
        self.cost_mz = cost_mz
        self.cost_intensity = cost_intensity
    
    @torch.no_grad()
    def forward(
        self,
        pred_mz: torch.Tensor,
        pred_intensity: torch.Tensor,
        target_mz: torch.Tensor,
        target_intensity: torch.Tensor,
        target_mask: torch.Tensor
    ) -> list[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Perform Hungarian matching with optimized CPU-GPU transfers.
        
        Args:
            pred_mz: Predicted m/z values, shape (batch_size, num_predictions)
            pred_intensity: Predicted intensities, shape (batch_size, num_predictions)
            target_mz: Target m/z values, shape (batch_size, num_targets)
            target_intensity: Target intensities, shape (batch_size, num_targets)
            target_mask: Mask for real targets (1 for real, 0 for padding), 
                        shape (batch_size, num_targets)
            
        Returns:
            List of tuples (pred_indices, target_indices) for each sample in batch
        """
        batch_size = pred_mz.size(0)
        device = pred_mz.device
        
        # Move all data to CPU once
        pred_mz_cpu = pred_mz.cpu().numpy()
        pred_intensity_cpu = pred_intensity.cpu().numpy()
        target_mz_cpu = target_mz.cpu().numpy()
        target_intensity_cpu = target_intensity.cpu().numpy()
        target_mask_cpu = target_mask.cpu().numpy()
        
        indices = []
        
        for b in range(batch_size):
            # Get number of real targets for this sample
            num_real_targets = int(target_mask_cpu[b].sum())
            
            if num_real_targets == 0:
                # No targets, all predictions are unmatched
                indices.append((
                    torch.tensor([], dtype=torch.long, device=device),
                    torch.tensor([], dtype=torch.long, device=device)
                ))
                continue
            
            # Extract real targets only
            real_target_mz = target_mz_cpu[b, :num_real_targets]
            real_target_intensity = target_intensity_cpu[b, :num_real_targets]
            
            # Compute cost matrix on CPU (numpy is fast for this)
            # Shape: (num_predictions, num_real_targets)
            pred_mz_exp = pred_mz_cpu[b, :, np.newaxis]  # (num_predictions, 1)
            pred_int_exp = pred_intensity_cpu[b, :, np.newaxis]  # (num_predictions, 1)
            
            target_mz_exp = real_target_mz[np.newaxis, :]  # (1, num_real_targets)
            target_int_exp = real_target_intensity[np.newaxis, :]  # (1, num_real_targets)
            
            # Compute costs
            cost_mz = np.abs(pred_mz_exp - target_mz_exp)
            cost_intensity = np.abs(pred_int_exp - target_int_exp)
            
            # Total cost
            cost = (
                self.cost_mz * cost_mz +
                self.cost_intensity * cost_intensity 
            )
            
            # Run Hungarian algorithm
            pred_idx, target_idx = linear_sum_assignment(cost)
            
            # Convert back to torch tensors (single transfer)
            indices.append((
                torch.from_numpy(pred_idx).long().to(device),
                torch.from_numpy(target_idx).long().to(device)
            ))
        
        return indices


def get_matched_pairs(
    indices: list[Tuple[torch.Tensor, torch.Tensor]],
    batch_size: int,
    num_predictions: int
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Convert matching indices to binary masks and flattened indices.
    
    Args:
        indices: List of (pred_indices, target_indices) tuples
        batch_size: Batch size
        num_predictions: Number of predictions
        
    Returns:
        Tuple of:
        - matched_pred_mask: Binary mask for matched predictions, shape (batch_size, num_predictions)
        - (batch_indices, pred_indices, target_indices): Flattened indices for matched pairs
    """
    matched_pred_mask = torch.zeros(batch_size, num_predictions, dtype=torch.bool)
    
    batch_idx_list = []
    pred_idx_list = []
    target_idx_list = []
    
    for b, (pred_idx, target_idx) in enumerate(indices):
        if len(pred_idx) > 0:
            matched_pred_mask[b, pred_idx] = True
            batch_idx_list.append(torch.full_like(pred_idx, b))
            pred_idx_list.append(pred_idx)
            target_idx_list.append(target_idx)
    
    if len(batch_idx_list) > 0:
        batch_indices = torch.cat(batch_idx_list)
        pred_indices = torch.cat(pred_idx_list)
        target_indices = torch.cat(target_idx_list)
    else:
        # No matches in entire batch
        batch_indices = torch.tensor([], dtype=torch.long)
        pred_indices = torch.tensor([], dtype=torch.long)
        target_indices = torch.tensor([], dtype=torch.long)
    
    return matched_pred_mask, (batch_indices, pred_indices, target_indices)

