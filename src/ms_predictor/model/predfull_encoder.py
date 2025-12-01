"""
PredFull CNN Encoder for peptide sequence processing.

Based on the architecture described in:
Liu, K.; Li, S.; Wang, L.; Ye, Y.; Tang, H.
Full-Spectrum Prediction of Peptides Tandem Mass Spectra using Deep Neural Network.
Analytical Chemistry 2020, 92, 4275-4283.

Architecture:
1. 8 parallel 1D convolutions with kernel sizes 2-9
2. Concatenate parallel conv outputs
3. 10 Squeeze-and-Excitation blocks
4. 3 Residual blocks (decoder)
5. Final projection to output dimension
"""

import torch
import torch.nn as nn
from typing import List

from .se_block import SqueezeExcitationBlock


class PredFullEncoder(nn.Module):
    """
    PredFull CNN-based encoder for peptide sequences.
    
    Uses parallel convolutions to capture multi-scale patterns,
    followed by SE blocks for channel attention and residual blocks
    for deep feature extraction.
    """
    
    def __init__(
        self,
        hidden_dim: int = 512,
        conv_kernel_sizes: List[int] = [2, 3, 4, 5, 6, 7, 8, 9],
        num_se_blocks: int = 10,
        se_reduction: int = 16,
        dropout: float = 0.1,
        activation: str = 'relu'
    ):
        """
        Initialize PredFull encoder.
        
        Args:
            hidden_dim: Dimension of hidden representations
            conv_kernel_sizes: List of kernel sizes for parallel convolutions
            num_se_blocks: Number of SE blocks to apply
            se_reduction: Reduction ratio for SE blocks
            dropout: Dropout rate
            activation: Activation function ('relu' or 'gelu')
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.conv_kernel_sizes = conv_kernel_sizes
        self.num_parallel_convs = len(conv_kernel_sizes)
        
        # 1. Parallel 1D convolutions with different kernel sizes
        # Each conv outputs hidden_dim channels
        # Use padding='same' to ensure all outputs have the same length
        self.parallel_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=k,
                    padding='same',  # Automatically compute padding to preserve length
                    bias=False
                ),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU() if activation == 'relu' else nn.GELU(),
                nn.Dropout(dropout)
            )
            for k in conv_kernel_sizes
        ])
        
        # After concatenation, we have hidden_dim * num_parallel_convs channels
        merged_channels = hidden_dim * self.num_parallel_convs
        
        # 2. Projection layer to reduce merged channels back to hidden_dim
        self.merge_projection = nn.Sequential(
            nn.Conv1d(merged_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU() if activation == 'relu' else nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 3. Squeeze-and-Excitation blocks
        self.se_blocks = nn.ModuleList([
            SqueezeExcitationBlock(
                channels=hidden_dim,
                reduction=se_reduction,
                activation=activation
            )
            for _ in range(num_se_blocks)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through PredFull encoder.
        
        Args:
            x: Input embeddings, shape (batch_size, seq_len, hidden_dim)
            
        Returns:
            Encoded representations, shape (batch_size, hidden_dim, seq_len) in channel-first format
        """
        batch_size, seq_len, hidden_dim = x.size()
        
        # Transpose for Conv1d: (batch_size, seq_len, hidden_dim) -> (batch_size, hidden_dim, seq_len)
        x = x.transpose(1, 2)
        
        # 1. Apply parallel convolutions
        parallel_outputs = []
        for conv in self.parallel_convs:
            conv_out = conv(x)
            parallel_outputs.append(conv_out)
        
        # 2. Concatenate parallel conv outputs along channel dimension
        # List of (batch_size, hidden_dim, seq_len) -> (batch_size, hidden_dim * num_parallel_convs, seq_len)
        merged = torch.cat(parallel_outputs, dim=1)
        
        # 3. Project back to hidden_dim
        x = self.merge_projection(merged)
        # x shape: (batch_size, hidden_dim, seq_len)
        
        # 4. Apply SE blocks
        # SE blocks perform channel-wise attention (scale input by learned weights)
        for se_block in self.se_blocks:
            x = se_block(x)
        
        # Output in channel-first format for decoder
        # x shape: (batch_size, hidden_dim, seq_len)
        return x

