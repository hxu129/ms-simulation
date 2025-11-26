"""
Embedding layers for the MS predictor model.
"""

import torch
import torch.nn as nn
import math


class AminoAcidEmbedding(nn.Module):
    """
    Embedding layer for amino acid tokens.
    """
    
    def __init__(self, vocab_size: int, hidden_dim: int, padding_idx: int = 0):
        """
        Initialize amino acid embedding.
        
        Args:
            vocab_size: Size of the amino acid vocabulary
            hidden_dim: Dimension of embeddings
            padding_idx: Index of padding token
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=padding_idx)
        self.hidden_dim = hidden_dim
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            tokens: Token indices, shape (batch_size, seq_len)
            
        Returns:
            Embeddings, shape (batch_size, seq_len, hidden_dim)
        """
        return self.embedding(tokens) * math.sqrt(self.hidden_dim)


class MetadataEmbedding(nn.Module):
    """
    Embedding layer for metadata (precursor m/z and charge).
    
    Creates special tokens for metadata that are prepended to the sequence.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        max_charge: int = 10,
        precursor_mz_bins: int = 100
    ):
        """
        Initialize metadata embedding.
        
        Args:
            hidden_dim: Dimension of embeddings
            max_charge: Maximum charge state to support
            precursor_mz_bins: Number of bins for precursor m/z discretization
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.precursor_mz_bins = precursor_mz_bins
        
        # Charge embedding
        self.charge_embedding = nn.Embedding(max_charge + 1, hidden_dim)
        
        # Precursor m/z embedding (using learned projection from continuous value)
        self.precursor_mz_projection = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
    
    def forward(
        self,
        precursor_mz: torch.Tensor,
        charge: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            precursor_mz: Normalized precursor m/z values, shape (batch_size,)
            charge: Charge states, shape (batch_size,)
            
        Returns:
            Metadata embeddings, shape (batch_size, 2, hidden_dim)
            First token is precursor m/z, second is charge
        """
        batch_size = precursor_mz.size(0)
        
        # Project precursor m/z
        precursor_emb = self.precursor_mz_projection(precursor_mz.unsqueeze(-1))
        precursor_emb = precursor_emb.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        
        # Embed charge
        charge_emb = self.charge_embedding(charge).unsqueeze(1)  # (batch_size, 1, hidden_dim)
        
        # Concatenate metadata tokens
        metadata_emb = torch.cat([precursor_emb, charge_emb], dim=1)  # (batch_size, 2, hidden_dim)
        
        return metadata_emb


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.
    """
    
    def __init__(self, hidden_dim: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Initialize positional encoding.
        
        Args:
            hidden_dim: Dimension of embeddings
            max_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encodings
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * (-math.log(10000.0) / hidden_dim))
        
        pe = torch.zeros(max_len, hidden_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input embeddings, shape (batch_size, seq_len, hidden_dim)
            
        Returns:
            Embeddings with positional encoding, same shape as input
        """
        seq_len = x.size(1)
        x = x + self.pe[:seq_len]
        return self.dropout(x)


class LearnableQueryEmbedding(nn.Module):
    """
    Learnable query embeddings for the decoder.
    
    These N queries will be used to predict N peaks.
    """
    
    def __init__(self, num_queries: int, hidden_dim: int):
        """
        Initialize learnable query embeddings.
        
        Args:
            num_queries: Number of query embeddings (N predictions)
            hidden_dim: Dimension of embeddings
        """
        super().__init__()
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        
        # Learnable query embeddings
        self.queries = nn.Parameter(torch.randn(num_queries, hidden_dim))
        
        # Positional encodings for queries
        self.query_pos = nn.Parameter(torch.randn(num_queries, hidden_dim))
    
    def forward(self, batch_size: int) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Query embeddings, shape (batch_size, num_queries, hidden_dim)
        """
        # Expand queries for batch
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)
        query_pos = self.query_pos.unsqueeze(0).expand(batch_size, -1, -1)
        
        return queries + query_pos


class InputEmbedding(nn.Module):
    """
    Combined input embedding for peptide sequence and metadata.
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        max_length: int,
        max_charge: int = 10,
        padding_idx: int = 0,
        dropout: float = 0.1
    ):
        """
        Initialize input embedding.
        
        Args:
            vocab_size: Size of amino acid vocabulary
            hidden_dim: Dimension of embeddings
            max_length: Maximum sequence length
            max_charge: Maximum charge state
            padding_idx: Padding token index
            dropout: Dropout rate
        """
        super().__init__()
        
        self.aa_embedding = AminoAcidEmbedding(vocab_size, hidden_dim, padding_idx)
        self.metadata_embedding = MetadataEmbedding(hidden_dim, max_charge)
        self.positional_encoding = PositionalEncoding(hidden_dim, max_length + 10, dropout)
    
    def forward(
        self,
        tokens: torch.Tensor,
        precursor_mz: torch.Tensor,
        charge: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            tokens: Token indices, shape (batch_size, seq_len)
            precursor_mz: Precursor m/z values, shape (batch_size,)
            charge: Charge states, shape (batch_size,)
            
        Returns:
            Combined embeddings, shape (batch_size, 2 + seq_len, hidden_dim)
            First 2 tokens are metadata (precursor m/z, charge), followed by sequence
        """
        # Get amino acid embeddings
        aa_emb = self.aa_embedding(tokens)  # (batch_size, seq_len, hidden_dim)
        
        # Get metadata embeddings
        metadata_emb = self.metadata_embedding(precursor_mz, charge)  # (batch_size, 2, hidden_dim)
        
        # Concatenate metadata and sequence
        combined_emb = torch.cat([metadata_emb, aa_emb], dim=1)  # (batch_size, 2 + seq_len, hidden_dim)
        
        # Add positional encoding
        combined_emb = self.positional_encoding(combined_emb)
        
        return combined_emb

