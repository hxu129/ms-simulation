"""Loss functions module."""

from .hungarian_matching import FastHungarianMatcher, get_matched_pairs
from .set_loss import SetPredictionLoss
from .cosine_loss import CosineSimilarityLoss, SpectralAngleLoss
from .binned_cosine_loss import BinnedCosineLoss

__all__ = [
    'FastHungarianMatcher',
    'get_matched_pairs',
    'SetPredictionLoss',
    'CosineSimilarityLoss',
    'SpectralAngleLoss',
    'BinnedCosineLoss',
]


