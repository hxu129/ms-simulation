"""
Residual Block for CNN encoder.

Based on the ResNet architecture:
He, K.; Zhang, X.; Ren, S.; Sun, J. 
Proceedings of the IEEE conference on computer vision and pattern recognition 2016.
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    1D Residual Block with skip connection.
    
    Applies two 1D convolutions with batch normalization and skip connection.
    If input and output dimensions differ, uses a 1x1 convolution for the skip connection.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        activation: str = 'relu',
        dropout: float = 0.1
    ):
        """
        Initialize residual block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Kernel size for convolutions
            stride: Stride for convolutions
            activation: Activation function ('relu' or 'gelu')
            dropout: Dropout rate
        """
        super().__init__()
        
        activation_fn = nn.GELU() if activation == 'gelu' else nn.ReLU()
        
        # Compute padding to preserve length when stride=1
        # For stride > 1, use standard padding
        if stride == 1:
            padding = 'same'
        else:
            padding = kernel_size // 2
        
        # First convolution block
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.activation1 = activation_fn
        self.dropout1 = nn.Dropout(dropout)
        
        # Second convolution block (always stride=1)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding='same',  # Always preserve length for second conv
            bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Skip connection
        if in_channels != out_channels or stride != 1:
            # Use 1x1 convolution to match dimensions
            self.skip_connection = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.skip_connection = nn.Identity()
        
        self.activation2 = activation_fn
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through residual block.
        
        Args:
            x: Input tensor, shape (batch_size, in_channels, length)
            
        Returns:
            Output tensor, shape (batch_size, out_channels, length)
        """
        identity = x
        
        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation1(out)
        out = self.dropout1(out)
        
        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Add skip connection
        identity = self.skip_connection(identity)
        out = out + identity
        
        # Final activation
        out = self.activation2(out)
        out = self.dropout2(out)
        
        return out

