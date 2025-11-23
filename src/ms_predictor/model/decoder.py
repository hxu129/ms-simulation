"""
Transformer Decoder with bidirectional attention for spectrum prediction.
"""

import torch
import torch.nn as nn
from typing import Optional

class DecoderLayer(nn.Module):
    """
    Custom decoder layer with explicit bidirectional attention.
    
    This is an alternative implementation that makes the bidirectional nature more explicit.
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
        Initialize bidirectional decoder layer.
        
        Args:
            hidden_dim: Dimension of hidden representations
            num_heads: Number of attention heads
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
        """
        super().__init__()
        if activation == 'gelu':
            activation_fn = nn.GELU()
        elif activation == 'relu':
            activation_fn = nn.ReLU()
        elif activation == 'tanh':
            activation_fn = nn.Tanh()
        else:
            raise ValueError(f"Invalid activation function: {activation}")
        
        # Self-attention (bidirectional)
        self.self_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Cross-attention (queries attend to encoder output)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, dim_feedforward),
            activation_fn,
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            tgt: Query embeddings, shape (batch_size, num_queries, hidden_dim)
            memory: Encoder output, shape (batch_size, src_len, hidden_dim)
            memory_key_padding_mask: Mask for encoder padding
            
        Returns:
            Output, shape (batch_size, num_queries, hidden_dim)
        """
        # Self-attention with residual connection (bidirectional)
        tgt2 = self.norm1(tgt)
        tgt2, _ = self.self_attn(tgt2, tgt2, tgt2, attn_mask=None)  # No causal mask
        tgt = tgt + self.dropout(tgt2)
        
        # Cross-attention with residual connection
        tgt2 = self.norm2(tgt)
        tgt2, _ = self.cross_attn(
            tgt2, memory, memory,
            key_padding_mask=memory_key_padding_mask
        )
        tgt = tgt + self.dropout(tgt2)
        
        # Feedforward with residual connection
        tgt2 = self.norm3(tgt)
        tgt2 = self.ffn(tgt2)
        tgt = tgt + tgt2
        
        return tgt


class Decoder(nn.Module):
    """
    Custom bidirectional decoder using explicit bidirectional layers.
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
        Initialize custom bidirectional decoder.
        
        Args:
            hidden_dim: Dimension of hidden representations
            num_layers: Number of decoder layers
            num_heads: Number of attention heads
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
        """
        super().__init__()
        
        self.layers = nn.ModuleList([
            DecoderLayer(hidden_dim, num_heads, dim_feedforward, dropout, activation)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            tgt: Query embeddings
            memory: Encoder output
            memory_key_padding_mask: Mask for encoder padding
            
        Returns:
            Decoded representations
        """
        output = tgt
        
        for layer in self.layers:
            output = layer(output, memory, memory_key_padding_mask)
        
        output = self.norm(output)
        
        return output

