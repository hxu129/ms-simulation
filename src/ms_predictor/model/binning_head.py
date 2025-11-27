"""
Binning head for predicting fixed-length binned spectrum.
"""

import torch
import torch.nn as nn
from typing import Optional


class BinningHead(nn.Module):
    """
    Prediction head that maps encoder embedding to binned spectrum.
    
    Takes a single embedding vector and outputs a fixed-length vector
    representing binned spectrum intensities.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_bins: int,
        intermediate_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        """
        Initialize binning head.
        
        Args:
            hidden_dim: Dimension of encoder output
            num_bins: Number of bins for spectrum representation
            intermediate_dim: Intermediate dimension for MLP (defaults to hidden_dim)
            dropout: Dropout rate
        """
        super().__init__()
        
        if intermediate_dim is None:
            intermediate_dim = hidden_dim
        
        # MLP: hidden_dim -> intermediate_dim -> num_bins
        self.binning_head = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, num_bins),
            nn.ReLU()  # Ensure non-negative output
        )
    
    def forward(self, encoder_embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through binning head.
        
        Args:
            encoder_embedding: Encoder embedding, shape (batch_size, hidden_dim)
            
        Returns:
            Binned spectrum prediction, shape (batch_size, num_bins)
            All values are non-negative due to final ReLU activation
        """
        binned_output = self.binning_head(encoder_embedding)
        return binned_output

