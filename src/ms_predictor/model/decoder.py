"""
Transformer Decoder with bidirectional attention for spectrum prediction.
"""

import torch
import torch.nn as nn
from typing import Optional


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder with bidirectional attention.
    
    Unlike standard autoregressive decoders, this decoder uses bidirectional self-attention
    among queries, allowing them to interact with each other and with the encoder output.
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
        Initialize Transformer Decoder.
        
        Args:
            hidden_dim: Dimension of hidden representations
            num_layers: Number of decoder layers
            num_heads: Number of attention heads
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
            activation: Activation function ('relu' or 'gelu')
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Create decoder layers
        # We use TransformerDecoderLayer which has:
        # 1. Self-attention (queries attend to each other) - we make this BIDIRECTIONAL
        # 2. Cross-attention (queries attend to encoder output)
        # 3. Feedforward network
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True  # Pre-LN for better training stability
        )
        
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim)
        )
    
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the decoder.
        
        Args:
            tgt: Query embeddings, shape (batch_size, num_queries, hidden_dim)
            memory: Encoder output (memory), shape (batch_size, src_len, hidden_dim)
            tgt_mask: Attention mask for target queries. For bidirectional attention,
                      this should be None (no causal masking)
            memory_key_padding_mask: Mask for encoder padding, shape (batch_size, src_len)
                                     True for positions to mask (padding), False otherwise
            
        Returns:
            Decoded representations, shape (batch_size, num_queries, hidden_dim)
        """
        # For bidirectional attention, we don't use causal mask (tgt_mask=None)
        # This allows all queries to attend to all other queries
        output = self.decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,  # None for bidirectional
            memory_key_padding_mask=memory_key_padding_mask
        )
        
        return output


class BidirectionalDecoderLayer(nn.Module):
    """
    Custom decoder layer with explicit bidirectional attention.
    
    This is an alternative implementation that makes the bidirectional nature more explicit.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float = 0.1
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
            nn.GELU(),
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


class CustomBidirectionalDecoder(nn.Module):
    """
    Custom bidirectional decoder using explicit bidirectional layers.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float = 0.1
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
            BidirectionalDecoderLayer(hidden_dim, num_heads, dim_feedforward, dropout)
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

