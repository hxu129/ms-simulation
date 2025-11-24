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


class ModificationTokenizer:
    """
    Tokenizer for peptide modifications.
    
    Converts modification strings to integer indices for embedding.
    Supports dynamic vocabulary building from data or fixed vocabulary.
    """
    
    def __init__(
        self, 
        modifications: Optional[List[str]] = None,
        no_mod_token: str = '<NO_MOD>',
        unk_mod_token: str = '<UNK_MOD>'
    ):
        """
        Initialize the modification tokenizer.
        
        Args:
            modifications: List of modification names (None for empty vocab to be built later)
            no_mod_token: Token for unmodified positions
            unk_mod_token: Token for unknown modifications
        """
        self.no_mod_token = no_mod_token
        self.unk_mod_token = unk_mod_token
        
        # Build vocabulary
        self.vocab = {
            no_mod_token: 0,
            unk_mod_token: 1,
        }
        
        # Add modifications to vocabulary if provided
        if modifications:
            for i, mod in enumerate(modifications, start=2):
                if mod not in self.vocab:
                    self.vocab[mod] = i
        
        # Reverse mapping
        self.idx_to_token = {idx: token for token, idx in self.vocab.items()}
        
        self.no_mod_idx = self.vocab[no_mod_token]
        self.unk_mod_idx = self.vocab[unk_mod_token]
    
    @property
    def vocab_size(self) -> int:
        """Return the size of the modification vocabulary."""
        return len(self.vocab)
    
    def add_modification(self, mod: str) -> int:
        """
        Add a modification to the vocabulary if not already present.
        
        Args:
            mod: Modification string
            
        Returns:
            Index of the modification
        """
        if mod not in self.vocab:
            self.vocab[mod] = len(self.vocab)
            self.idx_to_token[self.vocab[mod]] = mod
        return self.vocab[mod]
    
    def build_vocab_from_modifications(self, all_modifications: List[str]):
        """
        Build vocabulary from a list of unique modifications.
        
        Args:
            all_modifications: List of unique modification strings
        """
        for mod in all_modifications:
            if mod and mod not in self.vocab:
                self.add_modification(mod)
    
    def encode_modifications(
        self,
        modifications: List[Optional[str]],
        max_length: Optional[int] = None,
        padding: bool = True
    ) -> List[int]:
        """
        Convert modification list to token indices.
        
        Args:
            modifications: List of modification strings (None for unmodified positions)
            max_length: Maximum sequence length (pads or truncates if specified)
            padding: Whether to pad the sequence to max_length
            
        Returns:
            List of modification token indices
            
        Examples:
            >>> tokenizer = ModificationTokenizer(['ox', 'ph'])
            >>> tokenizer.encode_modifications([None, 'ox', None, 'ph'])
            [0, 2, 0, 3]
        """
        # Convert each modification to its index
        tokens = []
        for mod in modifications:
            if mod is None:
                tokens.append(self.no_mod_idx)
            else:
                tokens.append(self.vocab.get(mod, self.unk_mod_idx))
        
        if max_length is not None:
            if len(tokens) > max_length:
                # Truncate
                tokens = tokens[:max_length]
            elif padding and len(tokens) < max_length:
                # Pad with no_mod_idx
                tokens = tokens + [self.no_mod_idx] * (max_length - len(tokens))
        
        return tokens
    
    def decode_modifications(
        self,
        indices: List[int],
        skip_special_tokens: bool = True
    ) -> List[Optional[str]]:
        """
        Convert modification token indices back to modification strings.
        
        Args:
            indices: List of modification token indices
            skip_special_tokens: Whether to skip special tokens (NO_MOD, UNK_MOD) in output
            
        Returns:
            List of modification strings (None for unmodified positions)
        """
        modifications = []
        
        for idx in indices:
            if idx == self.no_mod_idx:
                modifications.append(None)
            elif skip_special_tokens and idx == self.unk_mod_idx:
                modifications.append(None)
            else:
                mod = self.idx_to_token.get(idx, self.unk_mod_token)
                if mod == self.no_mod_token:
                    modifications.append(None)
                else:
                    modifications.append(mod)
        
        return modifications
    
    def batch_encode_modifications(
        self,
        batch_modifications: List[List[Optional[str]]],
        max_length: Optional[int] = None,
        padding: bool = True
    ) -> List[List[int]]:
        """
        Encode a batch of modification sequences.
        
        Args:
            batch_modifications: List of modification lists
            max_length: Maximum sequence length
            padding: Whether to pad sequences
            
        Returns:
            List of modification token index lists
        """
        return [
            self.encode_modifications(mods, max_length, padding) 
            for mods in batch_modifications
        ]
    
    def batch_decode_modifications(
        self,
        batch_indices: List[List[int]],
        skip_special_tokens: bool = True
    ) -> List[List[Optional[str]]]:
        """
        Decode a batch of modification token indices.
        
        Args:
            batch_indices: List of modification token index lists
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            List of modification lists
        """
        return [
            self.decode_modifications(indices, skip_special_tokens) 
            for indices in batch_indices
        ]
