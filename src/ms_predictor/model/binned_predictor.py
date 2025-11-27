"""
Binned MS spectrum predictor using encoder and binning head.
"""

import torch
import torch.nn as nn
from typing import Optional

from .embeddings import InputEmbedding
from .encoder import Encoder
from .binning_head import BinningHead


class BinnedMSPredictor(nn.Module):
    """
    Binned spectrum prediction model.
    
    Architecture:
    1. Input embedding (peptide + metadata)
    2. Transformer encoder (processes peptide sequence)
    3. Extract last valid token embedding
    4. Binning head (maps embedding to binned spectrum)
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 512,
        num_encoder_layers: int = 6,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        num_bins: int = 1500,
        max_length: int = 50,
        max_charge: int = 10,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        """
        Initialize binned MS predictor model.
        
        Args:
            vocab_size: Size of amino acid vocabulary
            hidden_dim: Dimension of hidden representations
            num_encoder_layers: Number of encoder layers
            num_heads: Number of attention heads
            dim_feedforward: Dimension of feedforward networks
            num_bins: Number of bins for spectrum representation
            max_length: Maximum peptide sequence length
            max_charge: Maximum charge state
            dropout: Dropout rate
            activation: Activation function
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_bins = num_bins
        self.max_length = max_length
        
        # Input embedding (combines peptide tokens and metadata)
        self.input_embedding = InputEmbedding(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            max_length=max_length,
            max_charge=max_charge,
            padding_idx=0,
            dropout=dropout
        )
        
        # Transformer encoder
        self.encoder = Encoder(
            hidden_dim=hidden_dim,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation
        )
        
        # Binning head
        self.binning_head = BinningHead(
            hidden_dim=hidden_dim,
            num_bins=num_bins,
            intermediate_dim=hidden_dim,
            dropout=dropout
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        sequence_tokens: torch.Tensor,
        sequence_mask: torch.Tensor,
        precursor_mz: torch.Tensor,
        charge: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            sequence_tokens: Tokenized peptide sequences, shape (batch_size, seq_len)
            sequence_mask: Attention mask for sequences, shape (batch_size, seq_len)
                          True for real tokens, False for padding
            precursor_mz: Normalized precursor m/z values, shape (batch_size,)
            charge: Charge states, shape (batch_size,)
            
        Returns:
            Binned spectrum prediction, shape (batch_size, num_bins)
        """
        batch_size = sequence_tokens.size(0)
        device = sequence_tokens.device
        
        # 1. Embed input (peptide + metadata)
        embedded = self.input_embedding(sequence_tokens, precursor_mz, charge)
        # embedded shape: (batch_size, 2 + seq_len, hidden_dim)
        
        # Create padding mask for encoder
        # The input_embedding prepends 2 metadata tokens, so we need to adjust the mask
        metadata_mask = torch.ones(batch_size, 2, dtype=torch.bool, device=device)
        full_mask = torch.cat([metadata_mask, sequence_mask], dim=1)
        # Convert to padding mask: True for padding positions
        encoder_padding_mask = ~full_mask
        
        # 2. Encode with Transformer encoder
        encoder_output = self.encoder(embedded, src_key_padding_mask=encoder_padding_mask)
        # encoder_output shape: (batch_size, 2 + seq_len, hidden_dim)
        
        # 3. Extract last valid token embedding
        # sequence_mask shape: (batch_size, seq_len) - True for real tokens
        # We need to find the last valid sequence token (not padding)
        # The encoder output has structure: [metadata_token_0, metadata_token_1, seq_token_0, ..., seq_token_n, padding...]
        
        # Get the length of each sequence (number of real tokens, not including padding)
        seq_lengths = sequence_mask.sum(dim=1)  # (batch_size,)
        
        # Calculate indices for last valid token
        # +2 for metadata tokens, -1 for 0-indexing = +1
        last_valid_indices = seq_lengths + 1  # (batch_size,)
        
        # Extract embeddings at these indices
        batch_indices = torch.arange(batch_size, device=device)
        last_valid_embeddings = encoder_output[batch_indices, last_valid_indices]
        # last_valid_embeddings shape: (batch_size, hidden_dim)
        
        # 4. Predict binned spectrum
        binned_spectrum = self.binning_head(last_valid_embeddings)
        # binned_spectrum shape: (batch_size, num_bins)
        
        return binned_spectrum


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

