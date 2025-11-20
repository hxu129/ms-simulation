"""
Hungarian matching algorithm for set prediction.
"""

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from typing import Tuple


class HungarianMatcher(nn.Module):
    """
    Hungarian matcher for optimal bipartite matching between predictions and targets.
    
    Computes a cost matrix and finds the optimal assignment using the Hungarian algorithm.
    """
    
    def __init__(
        self,
        cost_mz: float = 1.0,
        cost_intensity: float = 1.0,
        cost_confidence: float = 1.0
    ):
        """
        Initialize Hungarian matcher.
        
        Args:
            cost_mz: Weight for m/z position cost
            cost_intensity: Weight for intensity cost
            cost_confidence: Weight for confidence cost (negative to encourage high confidence)
        """
        super().__init__()
        self.cost_mz = cost_mz
        self.cost_intensity = cost_intensity
        self.cost_confidence = cost_confidence
    
    @torch.no_grad()
    def forward(
        self,
        pred_mz: torch.Tensor,
        pred_intensity: torch.Tensor,
        pred_confidence: torch.Tensor,
        target_mz: torch.Tensor,
        target_intensity: torch.Tensor,
        target_mask: torch.Tensor
    ) -> list[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Perform Hungarian matching.
        
        Args:
            pred_mz: Predicted m/z values, shape (batch_size, num_predictions)
            pred_intensity: Predicted intensities, shape (batch_size, num_predictions)
            pred_confidence: Predicted confidences, shape (batch_size, num_predictions)
            target_mz: Target m/z values, shape (batch_size, num_targets)
            target_intensity: Target intensities, shape (batch_size, num_targets)
            target_mask: Mask for real targets (1 for real, 0 for padding), 
                        shape (batch_size, num_targets)
            
        Returns:
            List of tuples (pred_indices, target_indices) for each sample in batch
        """
        batch_size = pred_mz.size(0)
        num_predictions = pred_mz.size(1)
        
        indices = []
        
        for b in range(batch_size):
            # Get number of real targets for this sample
            num_real_targets = int(target_mask[b].sum().item())
            
            if num_real_targets == 0:
                # No targets, all predictions are unmatched
                indices.append((
                    torch.tensor([], dtype=torch.long, device=pred_mz.device),
                    torch.tensor([], dtype=torch.long, device=pred_mz.device)
                ))
                continue
            
            # Extract real targets only
            real_target_mz = target_mz[b, :num_real_targets]
            real_target_intensity = target_intensity[b, :num_real_targets]
            
            # Compute cost matrix: (num_predictions, num_real_targets)
            # Cost formula: λ_mz * |pred_mz - target_mz| + λ_int * |pred_int - target_int| - λ_conf * pred_conf
            
            # Expand dimensions for broadcasting
            pred_mz_expanded = pred_mz[b].unsqueeze(1)  # (num_predictions, 1)
            pred_intensity_expanded = pred_intensity[b].unsqueeze(1)  # (num_predictions, 1)
            pred_confidence_expanded = pred_confidence[b].unsqueeze(1)  # (num_predictions, 1)
            
            target_mz_expanded = real_target_mz.unsqueeze(0)  # (1, num_real_targets)
            target_intensity_expanded = real_target_intensity.unsqueeze(0)  # (1, num_real_targets)
            
            # Compute costs
            cost_mz = torch.abs(pred_mz_expanded - target_mz_expanded)
            cost_intensity = torch.abs(pred_intensity_expanded - target_intensity_expanded)
            
            # Total cost (we want to minimize this)
            cost = (
                self.cost_mz * cost_mz +
                self.cost_intensity * cost_intensity -
                self.cost_confidence * pred_confidence_expanded
            )
            
            # Convert to numpy for scipy's Hungarian algorithm
            cost_matrix = cost.cpu().numpy()
            
            # Run Hungarian algorithm
            pred_idx, target_idx = linear_sum_assignment(cost_matrix)
            
            # Convert back to torch tensors
            pred_idx = torch.tensor(pred_idx, dtype=torch.long, device=pred_mz.device)
            target_idx = torch.tensor(target_idx, dtype=torch.long, device=pred_mz.device)
            
            indices.append((pred_idx, target_idx))
        
        return indices


def get_matched_pairs(
    indices: list[Tuple[torch.Tensor, torch.Tensor]],
    batch_size: int,
    num_predictions: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert matching indices to binary masks.
    
    Args:
        indices: List of (pred_indices, target_indices) tuples
        batch_size: Batch size
        num_predictions: Number of predictions
        
    Returns:
        Tuple of:
        - matched_pred_mask: Binary mask for matched predictions, shape (batch_size, num_predictions)
        - matched_pred_idx: Indices of matched predictions (flattened)
        - matched_target_idx: Indices of matched targets (flattened)
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

