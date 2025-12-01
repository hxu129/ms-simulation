"""
PredFull-based Binned MS spectrum predictor.

Combines PredFull CNN encoder with PredFull decoder for spectrum prediction.
Follows the exact architecture from Liu et al. 2020.
"""

import torch
import torch.nn as nn
from typing import Optional, Literal

from .embeddings import InputEmbedding
from .predfull_encoder import PredFullEncoder
from .predfull_decoder import PredFullDecoder


class PredFullBinnedPredictor(nn.Module):
    """
    Binned spectrum prediction model using original PredFull architecture.
    
    Architecture (following Liu et al. 2020):
    1. Input embedding (peptide + metadata)
    2. PredFull CNN encoder (8 parallel convs → merge → 10 SE blocks)
    3. PredFull decoder (3 residual blocks → final 1D conv → aggregation)
    4. Output: binned spectrum (batch_size, num_bins)
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 512,
        conv_kernel_sizes: list = None,
        num_se_blocks: int = 10,
        num_residual_blocks: int = 3,
        se_reduction: int = 16,
        num_bins: int = 20000,
        max_length: int = 50,
        max_charge: int = 10,
        dropout: float = 0.1,
        activation: str = 'relu',
        aggregation: Literal['mean', 'sum', 'max'] = 'mean'
    ):
        """
        Initialize PredFull binned MS predictor model.
        
        Args:
            vocab_size: Size of amino acid vocabulary
            hidden_dim: Dimension of hidden representations
            conv_kernel_sizes: List of kernel sizes for parallel convolutions (default: [2-9])
            num_se_blocks: Number of SE blocks (default: 10 from paper)
            num_residual_blocks: Number of residual blocks in decoder (default: 3 from paper)
            se_reduction: Reduction ratio for SE blocks
            num_bins: Number of bins for spectrum representation (default: 20000 from paper)
            max_length: Maximum peptide sequence length
            max_charge: Maximum charge state
            dropout: Dropout rate
            activation: Activation function ('relu' or 'gelu')
            aggregation: Aggregation method over sequence dimension ('mean', 'sum', 'max')
        """
        super().__init__()
        
        if conv_kernel_sizes is None:
            conv_kernel_sizes = [2, 3, 4, 5, 6, 7, 8, 9]
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_bins = num_bins
        self.max_length = max_length
        self.aggregation = aggregation
        
        # Input embedding (combines peptide tokens and metadata)
        self.input_embedding = InputEmbedding(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            max_length=max_length,
            max_charge=max_charge,
            padding_idx=0,
            dropout=dropout
        )
        
        # PredFull CNN encoder (parallel convs + SE blocks)
        self.encoder = PredFullEncoder(
            hidden_dim=hidden_dim,
            conv_kernel_sizes=conv_kernel_sizes,
            num_se_blocks=num_se_blocks,
            se_reduction=se_reduction,
            dropout=dropout,
            activation=activation
        )
        
        # PredFull decoder (residual blocks + final conv)
        self.decoder = PredFullDecoder(
            hidden_dim=hidden_dim,
            num_bins=num_bins,
            num_residual_blocks=num_residual_blocks,
            activation=activation,
            dropout=dropout,
            aggregation=aggregation
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
        # 1. Embed input (peptide + metadata)
        embedded = self.input_embedding(sequence_tokens, precursor_mz, charge)
        # embedded shape: (batch_size, 2 + seq_len, hidden_dim)
        
        # 2. Encode with PredFull CNN encoder
        encoder_output = self.encoder(embedded)
        # encoder_output shape: (batch_size, hidden_dim, 2 + seq_len)
        
        # 3. Decode to spectrum bins
        binned_spectrum = self.decoder(encoder_output)
        # binned_spectrum shape: (batch_size, num_bins)
        
        return binned_spectrum


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

