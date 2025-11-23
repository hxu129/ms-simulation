"""
Amino acid tokenizer for peptide sequences.
"""

from typing import List, Optional


class AminoAcidTokenizer:
    """
    Tokenizer for converting amino acid sequences to integer indices.
    
    Supports 20 standard amino acids plus special tokens for padding and unknown residues.
    """
    
    # 20 standard amino acids
    AMINO_ACIDS = [
        'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
        'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'
    ]
    
    def __init__(self, pad_token: str = '<PAD>', unk_token: str = '<UNK>'):
        """
        Initialize the tokenizer.
        
        Args:
            pad_token: Token for padding sequences
            unk_token: Token for unknown amino acids
        """
        self.pad_token = pad_token
        self.unk_token = unk_token
        
        # Build vocabulary
        self.vocab = {
            pad_token: 0,
            unk_token: 1,
        }
        
        # Add amino acids to vocabulary
        for i, aa in enumerate(self.AMINO_ACIDS, start=2):
            self.vocab[aa] = i
        
        # Reverse mapping
        self.idx_to_token = {idx: token for token, idx in self.vocab.items()}
        
        self.pad_idx = self.vocab[pad_token]
        self.unk_idx = self.vocab[unk_token]
        
    @property
    def vocab_size(self) -> int:
        """Return the size of the vocabulary."""
        return len(self.vocab)
    
    def encode(
        self, 
        sequence: str, 
        max_length: Optional[int] = None,
        padding: bool = True
    ) -> List[int]:
        """
        Convert amino acid sequence to token indices.
        
        Args:
            sequence: Amino acid sequence string
            max_length: Maximum sequence length (pads or truncates if specified)
            padding: Whether to pad the sequence to max_length
            
        Returns:
            List of token indices
        """
        # Convert each character to its index
        tokens = [self.vocab.get(aa.upper(), self.unk_idx) for aa in sequence]
        
        if max_length is not None:
            if len(tokens) > max_length:
                # Truncate
                tokens = tokens[:max_length]
            elif padding and len(tokens) < max_length:
                # Pad
                tokens = tokens + [self.pad_idx] * (max_length - len(tokens))
        
        return tokens
    
    def decode(self, indices: List[int], skip_special_tokens: bool = True) -> str:
        """
        Convert token indices back to amino acid sequence.
        
        Args:
            indices: List of token indices
            skip_special_tokens: Whether to skip special tokens (PAD, UNK) in output
            
        Returns:
            Amino acid sequence string
        """
        sequence = []
        special_tokens = {self.pad_idx, self.unk_idx} if skip_special_tokens else set()
        
        for idx in indices:
            if idx not in special_tokens:
                sequence.append(self.idx_to_token.get(idx, self.unk_token))
        
        return ''.join(sequence)
    
    def batch_encode(
        self,
        sequences: List[str],
        max_length: Optional[int] = None,
        padding: bool = True
    ) -> List[List[int]]:
        """
        Encode a batch of sequences.
        
        Args:
            sequences: List of amino acid sequences
            max_length: Maximum sequence length
            padding: Whether to pad sequences
            
        Returns:
            List of token index lists
        """
        return [self.encode(seq, max_length, padding) for seq in sequences]
    
    def batch_decode(
        self,
        batch_indices: List[List[int]],
        skip_special_tokens: bool = True
    ) -> List[str]:
        """
        Decode a batch of token indices.
        
        Args:
            batch_indices: List of token index lists
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            List of amino acid sequences
        """
        return [self.decode(indices, skip_special_tokens) for indices in batch_indices]

