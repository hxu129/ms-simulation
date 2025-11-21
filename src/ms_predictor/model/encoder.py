"""
Transformer Encoder for processing peptide sequences.
"""

import torch
import torch.nn as nn
from typing import Optional


class EncoderLayer(nn.Module):
    """
    Custom encoder layer with explicit self-attention and feedforward network.
    
    This implementation makes the encoder architecture more explicit and customizable.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        """
        Initialize encoder layer.
        
        Args:
            hidden_dim: Dimension of hidden representations
            num_heads: Number of attention heads
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
            activation: Activation function ('relu' or 'gelu')
        """
        super().__init__()
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feedforward network
        activation_fn = nn.GELU() if activation == 'gelu' else nn.ReLU()
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, dim_feedforward),
            activation_fn,
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization (Pre-LN architecture)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            src: Source embeddings, shape (batch_size, seq_len, hidden_dim)
            src_key_padding_mask: Mask for padding tokens, shape (batch_size, seq_len)
                                  True for positions to mask (padding), False otherwise
            
        Returns:
            Output, shape (batch_size, seq_len, hidden_dim)
        """
        # Self-attention with residual connection
        src2 = self.norm1(src)
        src2, _ = self.self_attn(
            src2, src2, src2,
            key_padding_mask=src_key_padding_mask
        )
        src = src + self.dropout(src2)
        
        # Feedforward with residual connection
        src2 = self.norm2(src)
        src2 = self.ffn(src2)
        src = src + src2
        
        return src


class Encoder(nn.Module):
    """
    Custom Transformer Encoder using explicit encoder layers.
    
    Processes the peptide sequence (with metadata) to generate contextualized representations.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        """
        Initialize Transformer Encoder.
        
        Args:
            hidden_dim: Dimension of hidden representations
            num_layers: Number of encoder layers
            num_heads: Number of attention heads
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
            activation: Activation function ('relu' or 'gelu')
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Create encoder layers
        self.layers = nn.ModuleList([
            EncoderLayer(hidden_dim, num_heads, dim_feedforward, dropout, activation)
            for _ in range(num_layers)
        ])
        
        # Final layer normalization
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the encoder.
        
        Args:
            src: Input embeddings, shape (batch_size, seq_len, hidden_dim)
            src_key_padding_mask: Mask for padding tokens, shape (batch_size, seq_len)
                                  True for positions to mask (padding), False otherwise
            
        Returns:
            Encoded representations, shape (batch_size, seq_len, hidden_dim)
        """
        output = src
        
        # Pass through each encoder layer
        for layer in self.layers:
            output = layer(output, src_key_padding_mask)
        
        # Final layer normalization
        output = self.norm(output)
        
        return output
