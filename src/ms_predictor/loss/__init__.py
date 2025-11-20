"""Loss functions module."""

from .hungarian_matching import HungarianMatcher, get_matched_pairs
from .set_loss import SetPredictionLoss
from .cosine_loss import CosineSimilarityLoss, SpectralAngleLoss

__all__ = [
    'HungarianMatcher',
    'get_matched_pairs',
    'SetPredictionLoss',
    'CosineSimilarityLoss',
    'SpectralAngleLoss',
]

