"""
Amino acid tokenizer for peptide sequences.
"""

from typing import List, Optional


class AminoAcidTokenizer:
    """
    Tokenizer for converting amino acid sequences to integer indices.
    
    Supports 20 standard amino acids, modified amino acids, N-terminal modifications,
    plus special tokens for padding and unknown residues.
    """
    
    # All residue types including standard amino acids and modifications
    # Order matters: N-terminal modifications first, then modified AAs, then standard AAs
    RESIDUES = [
        # N-terminal modifications (4)
        '(acetyl)',
        '(carbamyl)',
        '(-nh3)',
        '(carbamyl)(-nh3)',
        # Modified amino acids (4)
        'C(carbamidomethyl)',
        'M(ox)',
        'N(deamide)',
        'Q(deamide)',
        # Standard amino acids (20)
        'G', 'A', 'S', 'P', 'V', 'T', 'C', 'L', 'I', 'N',
        'D', 'Q', 'K', 'E', 'M', 'H', 'F', 'R', 'Y', 'W'
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
        
        # Add all residues to vocabulary (including modified amino acids and N-terminal modifications)
        for i, residue in enumerate(self.RESIDUES, start=2):
            self.vocab[residue] = i
        
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
        
        Handles modified amino acids (e.g., M(ox), C(carbamidomethyl)) and 
        N-terminal modifications (e.g., (acetyl), (carbamyl)).
        
        Args:
            sequence: Amino acid sequence string (e.g., "(acetyl)M(ox)PEPTIDE")
            max_length: Maximum sequence length (pads or truncates if specified)
            padding: Whether to pad the sequence to max_length
            
        Returns:
            List of token indices
        """
        tokens = []
        i = 0
        
        while i < len(sequence):
            matched = False
            
            # Try to match the longest valid token starting at position i
            # Check from longest to shortest possible matches
            for length in range(min(20, len(sequence) - i), 0, -1):
                candidate = sequence[i:i+length]
                
                if candidate in self.vocab:
                    tokens.append(self.vocab[candidate])
                    i += length
                    matched = True
                    break
            
            if not matched:
                # If no match found, treat as single character (possibly unknown)
                char = sequence[i].upper()
                tokens.append(self.vocab.get(char, self.unk_idx))
                i += 1
        
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
        
        Correctly reconstructs sequences with modifications (e.g., M(ox), (acetyl)).
        
        Args:
            indices: List of token indices
            skip_special_tokens: Whether to skip special tokens (PAD, UNK) in output
            
        Returns:
            Amino acid sequence string with modifications preserved
        """
        sequence = []
        special_tokens = {self.pad_idx, self.unk_idx} if skip_special_tokens else set()
        
        for idx in indices:
            if idx not in special_tokens:
                token = self.idx_to_token.get(idx, self.unk_token)
                sequence.append(token)
        
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

