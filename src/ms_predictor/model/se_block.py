"""
Squeeze-and-Excitation Block for channel-wise attention.

Based on:
Hu, J.; Shen, L.; Albanie, S.; Sun, G.; Wu, E. 
Squeeze-and-Excitation Networks. 
IEEE transactions on pattern analysis and machine intelligence 2019.
"""

import torch
import torch.nn as nn


class SqueezeExcitationBlock(nn.Module):
    """
    Squeeze-and-Excitation Block for channel-wise attention.
    
    The SE block performs:
    1. Global average pooling to squeeze spatial information
    2. Two FC layers to capture channel-wise dependencies
    3. Sigmoid activation to generate channel attention weights
    4. Scale input features by attention weights
    """
    
    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        activation: str = 'relu'
    ):
        """
        Initialize SE block.
        
        Args:
            channels: Number of input channels
            reduction: Reduction ratio for bottleneck (default: 16)
            activation: Activation function ('relu' or 'gelu')
        """
        super().__init__()
        
        self.channels = channels
        reduced_channels = max(channels // reduction, 1)
        
        # Squeeze: global average pooling (will be done in forward)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Excitation: two FC layers
        activation_fn = nn.GELU() if activation == 'gelu' else nn.ReLU()
        
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels),
            activation_fn,
            nn.Linear(reduced_channels, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through SE block.
        
        Args:
            x: Input tensor, shape (batch_size, channels, length)
            
        Returns:
            Output tensor with channel-wise attention applied, same shape as input
        """
        batch_size, channels, length = x.size()
        
        # Squeeze: global average pooling
        # (batch_size, channels, length) -> (batch_size, channels, 1)
        squeezed = self.avg_pool(x)
        # (batch_size, channels, 1) -> (batch_size, channels)
        squeezed = squeezed.view(batch_size, channels)
        
        # Excitation: FC layers with sigmoid
        # (batch_size, channels) -> (batch_size, channels)
        attention = self.fc(squeezed)
        
        # Reshape for broadcasting
        # (batch_size, channels) -> (batch_size, channels, 1)
        attention = attention.view(batch_size, channels, 1)
        
        # Scale input by attention weights
        output = x * attention
        
        return output

