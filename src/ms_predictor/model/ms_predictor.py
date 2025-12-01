"""
Main MS spectrum predictor model.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

from .embeddings import InputEmbedding, LearnableQueryEmbedding
from .encoder import Encoder
from .predfull_encoder import PredFullEncoder
from .decoder import Decoder
from .heads import PredictionHeads


class MSPredictor(nn.Module):
    """
    Main model for MS spectrum prediction.
    
    Architecture:
    1. Input embedding (peptide + metadata)
    2. Transformer encoder (processes peptide sequence)
    3. Learnable query embeddings (N queries)
    4. Transformer decoder with bidirectional attention
    5. Prediction heads (m/z, intensity, confidence)
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 512,
        encoder_type: str = 'transformer',
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        num_predictions: int = 100,
        max_length: int = 50,
        max_charge: int = 10,
        dropout: float = 0.1,
        activation: str = 'gelu',
        conv_kernel_sizes: list = None,
        num_se_blocks: int = 10,
        se_reduction: int = 16
    ):
        """
        Initialize MS predictor model.
        
        Args:
            vocab_size: Size of amino acid vocabulary
            hidden_dim: Dimension of hidden representations
            encoder_type: Type of encoder ('transformer' or 'predfull')
            num_encoder_layers: Number of encoder layers (for transformer encoder)
            num_decoder_layers: Number of decoder layers
            num_heads: Number of attention heads (for transformer encoder)
            dim_feedforward: Dimension of feedforward networks (for transformer encoder)
            num_predictions: Number of predictions to make (N)
            max_length: Maximum peptide sequence length
            max_charge: Maximum charge state
            dropout: Dropout rate
            activation: Activation function
            conv_kernel_sizes: List of kernel sizes for parallel convolutions (for predfull encoder)
            num_se_blocks: Number of SE blocks (for predfull encoder)
            se_reduction: Reduction ratio for SE blocks (for predfull encoder)
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.encoder_type = encoder_type
        self.num_predictions = num_predictions
        self.max_length = max_length
        
        # Set default conv_kernel_sizes if not provided
        if conv_kernel_sizes is None:
            conv_kernel_sizes = [2, 3, 4, 5, 6, 7, 8, 9]
        
        # Input embedding (combines peptide tokens and metadata)
        self.input_embedding = InputEmbedding(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            max_length=max_length,
            max_charge=max_charge,
            padding_idx=0,
            dropout=dropout
        )
        
        # Encoder (Transformer or PredFull)
        if encoder_type == 'transformer':
            self.encoder = Encoder(
                hidden_dim=hidden_dim,
                num_layers=num_encoder_layers,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation
            )
        elif encoder_type == 'predfull':
            self.encoder = PredFullEncoder(
                hidden_dim=hidden_dim,
                conv_kernel_sizes=conv_kernel_sizes,
                num_se_blocks=num_se_blocks,
                se_reduction=se_reduction,
                dropout=dropout,
                activation=activation
            )
        else:
            raise ValueError(f"Invalid encoder_type: {encoder_type}. Must be 'transformer' or 'predfull'.")
        
        # Learnable query embeddings
        self.query_embedding = LearnableQueryEmbedding(
            num_queries=num_predictions,
            hidden_dim=hidden_dim
        )
        
        # Transformer decoder (bidirectional)
        self.decoder = Decoder(
            hidden_dim=hidden_dim,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation
        )
        
        # Prediction heads
        self.prediction_heads = PredictionHeads(
            hidden_dim=hidden_dim,
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            sequence_tokens: Tokenized peptide sequences, shape (batch_size, seq_len)
            sequence_mask: Attention mask for sequences, shape (batch_size, seq_len)
                          True for real tokens, False for padding
            precursor_mz: Normalized precursor m/z values, shape (batch_size,)
            charge: Charge states, shape (batch_size,)
            
        Returns:
            Tuple of (mz_pred, intensity_pred, confidence_pred)
            - mz_pred: Predicted m/z values in [0, 1], shape (batch_size, num_predictions)
            - intensity_pred: Predicted intensities in [0, 1], shape (batch_size, num_predictions)
            - confidence_pred: Confidence scores in [0, 1], shape (batch_size, num_predictions)
        """
        batch_size = sequence_tokens.size(0)
        
        # 1. Embed input (peptide + metadata)
        embedded = self.input_embedding(sequence_tokens, precursor_mz, charge)
        # embedded shape: (batch_size, 2 + seq_len, hidden_dim)
        
        # Create padding mask for encoder
        # The input_embedding prepends 2 metadata tokens, so we need to adjust the mask
        metadata_mask = torch.ones(batch_size, 2, dtype=torch.bool, device=sequence_tokens.device)
        full_mask = torch.cat([metadata_mask, sequence_mask], dim=1)
        # Convert to padding mask: True for padding positions
        encoder_padding_mask = ~full_mask
        
        # 2. Encode with encoder (Transformer or PredFull)
        if self.encoder_type == 'transformer':
            encoder_output = self.encoder(embedded, src_key_padding_mask=encoder_padding_mask)
            # encoder_output shape: (batch_size, 2 + seq_len, hidden_dim)
        elif self.encoder_type == 'predfull':
            encoder_output = self.encoder(embedded)
            # encoder_output shape: (batch_size, hidden_dim, 2 + seq_len)
            # Transpose to (batch_size, 2 + seq_len, hidden_dim) for decoder
            encoder_output = encoder_output.transpose(1, 2)
        else:
            raise ValueError(f"Invalid encoder_type: {self.encoder_type}")
        
        # 3. Get learnable query embeddings
        queries = self.query_embedding(batch_size)
        # queries shape: (batch_size, num_predictions, hidden_dim)
        
        # 4. Decode with Transformer decoder (bidirectional)
        decoder_output = self.decoder(
            tgt=queries,
            memory=encoder_output,
            memory_key_padding_mask=encoder_padding_mask
        )
        # decoder_output shape: (batch_size, num_predictions, hidden_dim)
        
        # 5. Predict m/z, intensity, and confidence
        mz_pred, intensity_pred, confidence_pred = self.prediction_heads(decoder_output)
        
        return mz_pred, intensity_pred, confidence_pred
    
    def predict(
        self,
        sequence_tokens: torch.Tensor,
        sequence_mask: torch.Tensor,
        precursor_mz: torch.Tensor,
        charge: torch.Tensor,
        confidence_threshold: float = 0.5,
        max_mz: float = 2000.0
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions and filter by confidence threshold.
        
        Args:
            sequence_tokens: Tokenized peptide sequences
            sequence_mask: Attention mask for sequences
            precursor_mz: Normalized precursor m/z values
            charge: Charge states
            confidence_threshold: Threshold for filtering predictions
            max_mz: Maximum m/z for denormalization
            
        Returns:
            Dictionary containing filtered predictions
        """
        # Get predictions
        mz_pred, intensity_pred, confidence_pred = self.forward(
            sequence_tokens, sequence_mask, precursor_mz, charge
        )
        
        # Denormalize m/z
        mz_denorm = mz_pred * max_mz
        
        # Filter by confidence threshold
        confidence_mask = confidence_pred > confidence_threshold
        
        return {
            'mz': mz_denorm,
            'intensity': intensity_pred,
            'confidence': confidence_pred,
            'mask': confidence_mask
        }


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

