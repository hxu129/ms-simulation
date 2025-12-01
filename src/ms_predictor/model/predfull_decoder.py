"""
PredFull Decoder for spectrum prediction.

Based on the architecture described in:
Liu, K.; Li, S.; Wang, L.; Ye, Y.; Tang, H.
Full-Spectrum Prediction of Peptides Tandem Mass Spectra using Deep Neural Network.
Analytical Chemistry 2020, 92, 4275-4283.

The decoder consists of:
1. Three residual blocks
2. Final 1D convolutional layer that projects to num_bins
3. Aggregation over sequence dimension
"""

import torch
import torch.nn as nn
from typing import Literal

from .residual_block import ResidualBlock


class PredFullDecoder(nn.Module):
    """
    PredFull decoder for converting encoded sequence to spectrum bins.
    
    As described in the paper:
    "Three subsequently residual blocks and the last 1-dimensional convolutional 
    layer work as the decoder, which decodes the previous tensor into the final 
    prediction vector of length 20,000 representing the final MS/MS spectrum."
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_bins: int,
        num_residual_blocks: int = 3,
        activation: str = 'relu',
        dropout: float = 0.1,
        aggregation: Literal['mean', 'sum', 'max'] = 'mean'
    ):
        """
        Initialize PredFull decoder.
        
        Args:
            hidden_dim: Dimension of hidden representations from encoder
            num_bins: Number of output bins for spectrum
            num_residual_blocks: Number of residual blocks (default: 3 from paper)
            activation: Activation function ('relu' or 'gelu')
            dropout: Dropout rate
            aggregation: How to aggregate over sequence dimension ('mean', 'sum', 'max')
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_bins = num_bins
        self.aggregation = aggregation
        
        # 1. Residual blocks (decoder part)
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                activation=activation,
                dropout=dropout
            )
            for _ in range(num_residual_blocks)
        ])
        
        # 2. Final 1D convolutional layer
        # Projects from hidden_dim channels to num_bins channels
        # Using kernel_size=1 for point-wise convolution
        self.final_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=hidden_dim,
                out_channels=num_bins,
                kernel_size=1,
                bias=True
            ),
            nn.ReLU()  # Ensure non-negative spectrum intensities
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through PredFull decoder.
        
        Args:
            x: Encoded representations from encoder, shape (batch_size, hidden_dim, seq_len)
            
        Returns:
            Spectrum predictions, shape (batch_size, num_bins)
        """
        # x shape: (batch_size, hidden_dim, seq_len)
        
        # 1. Apply residual blocks
        for residual_block in self.residual_blocks:
            x = residual_block(x)
        # x shape: (batch_size, hidden_dim, seq_len)
        
        # 2. Final 1D convolution to project to num_bins
        x = self.final_conv(x)
        # x shape: (batch_size, num_bins, seq_len)
        
        # 3. Aggregate over sequence dimension
        if self.aggregation == 'mean':
            # Global average pooling over sequence
            output = x.mean(dim=2)  # (batch_size, num_bins)
        elif self.aggregation == 'sum':
            # Sum over sequence
            output = x.sum(dim=2)  # (batch_size, num_bins)
        elif self.aggregation == 'max':
            # Max pooling over sequence
            output, _ = x.max(dim=2)  # (batch_size, num_bins)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")
        
        return output

