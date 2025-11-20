"""
Transformer Encoder for processing peptide sequences.
"""

import torch
import torch.nn as nn
from typing import Optional


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder for processing peptide sequences.
    
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
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True  # Pre-LN for better training stability
        )
        
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim)
        )
    
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
        # The encoder expects True for positions to mask
        output = self.encoder(src, src_key_padding_mask=src_key_padding_mask)
        
        return output

