"""
Set prediction loss using Hungarian matching.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

# Matching algorithm options (benchmarked on batch_size=128):
# 1. FastHungarianMatcher (RECOMMENDED): 10.5ms, exact, 3.8x faster than original
from .hungarian_matching import FastHungarianMatcher as HungarianMatcher, get_matched_pairs
# 2. HungarianMatcher (Original): 39.7ms, exact but slow (multiple CPU-GPU transfers)
# from .hungarian_matching import HungarianMatcher, get_matched_pairs
# 3. GreedyMatcher: 278ms, slow + only 74.7% match quality (NOT recommended)
# from .hungarian_matching_greedy import GreedyMatcher as HungarianMatcher, get_matched_pairs


class SetPredictionLoss(nn.Module):
    """
    Set prediction loss for mass spectrum prediction.
    
    Uses Hungarian matching to find optimal assignment between predictions and targets,
    then computes losses for matched and unmatched predictions.
    """
    
    def __init__(
        self,
        cost_mz: float = 1.0,
        cost_intensity: float = 1.0,
        cost_confidence: float = 1.0,
        loss_mz_weight: float = 1.0,
        loss_intensity_weight: float = 1.0,
        loss_confidence_weight: float = 1.0,
        background_confidence_weight: float = 0.1
    ):
        """
        Initialize set prediction loss.
        
        Args:
            cost_mz: Weight for m/z cost in Hungarian matching
            cost_intensity: Weight for intensity cost in Hungarian matching
            cost_confidence: Weight for confidence cost in Hungarian matching
            loss_mz_weight: Weight for m/z loss
            loss_intensity_weight: Weight for intensity loss
            loss_confidence_weight: Weight for confidence loss (matched predictions)
            background_confidence_weight: Weight for background confidence loss (unmatched predictions)
        """
        super().__init__()
        
        self.matcher = HungarianMatcher(cost_mz, cost_intensity)
        
        self.loss_mz_weight = loss_mz_weight
        self.loss_intensity_weight = loss_intensity_weight
        self.loss_confidence_weight = loss_confidence_weight
        self.background_confidence_weight = background_confidence_weight
    
    def forward(
        self,
        pred_mz: torch.Tensor,
        pred_intensity: torch.Tensor,
        pred_confidence_logits: torch.Tensor,
        target_mz: torch.Tensor,
        target_intensity: torch.Tensor,
        target_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute set prediction loss.
        
        Args:
            pred_mz: Predicted m/z values, shape (batch_size, num_predictions)
            pred_intensity: Predicted intensities, shape (batch_size, num_predictions)
            pred_confidence: Predicted confidences, shape (batch_size, num_predictions)
            target_mz: Target m/z values, shape (batch_size, num_targets)
            target_intensity: Target intensities, shape (batch_size, num_targets)
            target_mask: Mask for real targets, shape (batch_size, num_targets)
            
        Returns:
            Dictionary of losses
        """
        batch_size = pred_mz.size(0)
        num_predictions = pred_mz.size(1)
        
        # Perform Hungarian matching
        indices = self.matcher(
            pred_mz, pred_intensity, pred_confidence_logits.sigmoid(),
            target_mz, target_intensity, target_mask
        )
        
        # Get matched pairs
        matched_pred_mask, (batch_idx, pred_idx, target_idx) = get_matched_pairs(
            indices, batch_size, num_predictions
        )
        
        # Initialize losses
        loss_mz = torch.tensor(0.0, device=pred_mz.device)
        loss_intensity = torch.tensor(0.0, device=pred_mz.device)
        loss_confidence_matched = torch.tensor(0.0, device=pred_mz.device)
        loss_confidence_background = torch.tensor(0.0, device=pred_mz.device)
        
        num_matched = len(pred_idx)
        
        # Compute losses for matched predictions
        if num_matched > 0:
            # Get matched predictions and targets
            matched_pred_mz = pred_mz[batch_idx, pred_idx]
            matched_pred_intensity = pred_intensity[batch_idx, pred_idx]
            # TODO: best practice is to use the sigmoid after the model
            matched_pred_confidence_logits = pred_confidence_logits[batch_idx, pred_idx]
            
            matched_target_mz = target_mz[batch_idx, target_idx]
            matched_target_intensity = target_intensity[batch_idx, target_idx]
            
            # L1 loss for m/z
            loss_mz = F.l1_loss(matched_pred_mz, matched_target_mz)
            
            # L1 loss for intensity
            loss_intensity = F.l1_loss(matched_pred_intensity, matched_target_intensity)
            
            # Binary cross-entropy for confidence (target = 1 for matched predictions)
            target_confidence_matched = torch.ones_like(matched_pred_confidence_logits)
            loss_confidence_matched = F.binary_cross_entropy_with_logits(
                matched_pred_confidence_logits,
                target_confidence_matched
            )
        
        # Compute loss for unmatched predictions (background)
        unmatched_pred_mask = ~matched_pred_mask
        num_unmatched = unmatched_pred_mask.sum().item()
        
        if num_unmatched > 0:
            unmatched_pred_confidence_logits = pred_confidence_logits[unmatched_pred_mask]
            
            # Binary cross-entropy for confidence (target = 0 for unmatched predictions)
            target_confidence_background = torch.zeros_like(unmatched_pred_confidence_logits)
            loss_confidence_background = F.binary_cross_entropy_with_logits(
                unmatched_pred_confidence_logits,
                target_confidence_background
            )
        
        # Total loss
        total_loss = (
            self.loss_mz_weight * loss_mz +
            self.loss_intensity_weight * loss_intensity +
            self.loss_confidence_weight * loss_confidence_matched +
            self.background_confidence_weight * loss_confidence_background
        )
        
        return {
            'loss': total_loss,
            'loss_mz': loss_mz,
            'loss_intensity': loss_intensity,
            'loss_confidence_matched': loss_confidence_matched,
            'loss_confidence_background': loss_confidence_background,
            'num_matched': torch.tensor(num_matched, device=pred_mz.device, dtype=torch.float32),
            'num_unmatched': torch.tensor(num_unmatched, device=pred_mz.device, dtype=torch.float32),
            'matched_indices': (batch_idx, pred_idx, target_idx)
        }

