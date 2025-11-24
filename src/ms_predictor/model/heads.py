"""
Prediction heads for m/z, intensity, and confidence.
"""

import torch
import torch.nn as nn
from typing import Optional


class PredictionHeads(nn.Module):
    """
    Three prediction heads for predicting m/z, intensity, and confidence.
    
    Each head takes the decoder output and produces predictions in [0, 1] range.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        """
        Initialize prediction heads.
        
        Args:
            hidden_dim: Dimension of decoder output
            intermediate_dim: Intermediate dimension for MLP (defaults to hidden_dim)
            dropout: Dropout rate
        """
        super().__init__()
        
        if intermediate_dim is None:
            intermediate_dim = hidden_dim
        
        # Position head (m/z prediction)
        self.position_head = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, 1),
            nn.Softplus()  
        )
        
        # Intensity head
        self.intensity_head = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, 1),
            nn.Softplus()  
        )
        
        # Confidence head (probability that this is a real peak)
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, 1),
            # nn.Sigmoid()  # Output in [0, 1] # TODO: best practice is to use the sigmoid after the model
        )
    
    def forward(self, decoder_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through all prediction heads.
        
        Args:
            decoder_output: Decoder output, shape (batch_size, num_queries, hidden_dim)
            
        Returns:
            Tuple of (mz_pred, intensity_pred, confidence_pred)
            - mz_pred: Predicted m/z values in [0, 1], shape (batch_size, num_queries)
            - intensity_pred: Predicted intensities in [0, 1], shape (batch_size, num_queries)
            - confidence_pred: Confidence scores in [0, 1], shape (batch_size, num_queries)
        """
        # Pass through each head
        mz_pred = self.position_head(decoder_output).squeeze(-1)  # (batch_size, num_queries)
        intensity_pred = self.intensity_head(decoder_output).squeeze(-1)  # (batch_size, num_queries)
        confidence_pred = self.confidence_head(decoder_output).squeeze(-1)  # (batch_size, num_queries)
        
        return mz_pred, intensity_pred, confidence_pred


# class SharedBackbonePredictionHeads(nn.Module):
#     """
#     Prediction heads with a shared backbone before splitting into specific heads.
    
#     This can help with learning shared representations before specializing.
#     """
    
#     def __init__(
#         self,
#         hidden_dim: int,
#         shared_dim: int,
#         dropout: float = 0.1
#     ):
#         """
#         Initialize prediction heads with shared backbone.
        
#         Args:
#             hidden_dim: Dimension of decoder output
#             shared_dim: Dimension of shared representation
#             dropout: Dropout rate
#         """
#         super().__init__()
        
#         # Shared backbone
#         self.shared_backbone = nn.Sequential(
#             nn.Linear(hidden_dim, shared_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#         )
        
#         # Position head
#         self.position_head = nn.Sequential(
#             nn.Linear(shared_dim, shared_dim // 2),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(shared_dim // 2, 1),
#             nn.Sigmoid()
#         )
        
#         # Intensity head
#         self.intensity_head = nn.Sequential(
#             nn.Linear(shared_dim, shared_dim // 2),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(shared_dim // 2, 1),
#             nn.Sigmoid()
#         )
        
#         # Confidence head
#         self.confidence_head = nn.Sequential(
#             nn.Linear(shared_dim, shared_dim // 2),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(shared_dim // 2, 1),
#             nn.Sigmoid()
#         )
    
#     def forward(self, decoder_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         """
#         Forward pass through shared backbone and prediction heads.
        
#         Args:
#             decoder_output: Decoder output, shape (batch_size, num_queries, hidden_dim)
            
#         Returns:
#             Tuple of (mz_pred, intensity_pred, confidence_pred)
#         """
#         # Pass through shared backbone
#         shared_repr = self.shared_backbone(decoder_output)
        
#         # Pass through each head
#         mz_pred = self.position_head(shared_repr).squeeze(-1)
#         intensity_pred = self.intensity_head(shared_repr).squeeze(-1)
#         confidence_pred = self.confidence_head(shared_repr).squeeze(-1)
        
#         return mz_pred, intensity_pred, confidence_pred

