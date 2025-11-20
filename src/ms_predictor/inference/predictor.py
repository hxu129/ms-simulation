"""
Inference pipeline for MS spectrum prediction.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional

from ..model.ms_predictor import MSPredictor
from ..data.tokenizer import AminoAcidTokenizer
from ..data.preprocessing import SpectrumPreprocessor


class SpectrumPredictor:
    """
    Inference pipeline for predicting mass spectra from peptide sequences.
    """
    
    def __init__(
        self,
        model: MSPredictor,
        tokenizer: Optional[AminoAcidTokenizer] = None,
        confidence_threshold: float = 0.5,
        max_mz: float = 2000.0,
        device: Optional[torch.device] = None
    ):
        """
        Initialize spectrum predictor.
        
        Args:
            model: Trained MSPredictor model
            tokenizer: Amino acid tokenizer (creates new one if None)
            confidence_threshold: Threshold for filtering predictions
            max_mz: Maximum m/z value for denormalization
            device: Device to use (defaults to CUDA if available)
        """
        self.model = model
        self.tokenizer = tokenizer or AminoAcidTokenizer()
        self.confidence_threshold = confidence_threshold
        self.max_mz = max_mz
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        
        self.model.to(self.device)
        self.model.eval()
    
    @torch.no_grad()
    def predict_single(
        self,
        sequence: str,
        precursor_mz: float,
        charge: int,
        max_intensity: float = 1.0
    ) -> Dict[str, np.ndarray]:
        """
        Predict spectrum for a single peptide sequence.
        
        Args:
            sequence: Amino acid sequence
            precursor_mz: Precursor m/z value
            charge: Charge state
            max_intensity: Maximum intensity for denormalization
            
        Returns:
            Dictionary containing:
                - mz: Array of m/z values
                - intensity: Array of intensity values
                - confidence: Array of confidence scores
        """
        # Tokenize sequence
        tokens = self.tokenizer.encode(
            sequence,
            max_length=self.model.max_length,
            padding=True
        )
        tokens_tensor = torch.LongTensor([tokens]).to(self.device)
        
        # Create mask
        mask = tokens_tensor != self.tokenizer.pad_idx
        
        # Normalize precursor m/z
        norm_precursor_mz = torch.tensor([precursor_mz / self.max_mz], dtype=torch.float32).to(self.device)
        charge_tensor = torch.tensor([charge], dtype=torch.long).to(self.device)
        
        # Predict
        pred_mz, pred_intensity, pred_confidence = self.model(
            tokens_tensor,
            mask,
            norm_precursor_mz,
            charge_tensor
        )
        
        # Move to CPU and convert to numpy
        pred_mz = pred_mz[0].cpu().numpy()
        pred_intensity = pred_intensity[0].cpu().numpy()
        pred_confidence = pred_confidence[0].cpu().numpy()
        
        # Denormalize m/z
        pred_mz = pred_mz * self.max_mz
        
        # Denormalize intensity
        pred_intensity = pred_intensity * max_intensity
        
        # Filter by confidence threshold
        confident_mask = pred_confidence > self.confidence_threshold
        
        return {
            'mz': pred_mz[confident_mask],
            'intensity': pred_intensity[confident_mask],
            'confidence': pred_confidence[confident_mask]
        }
    
    @torch.no_grad()
    def predict_batch(
        self,
        sequences: List[str],
        precursor_mz_list: List[float],
        charge_list: List[int],
        max_intensity_list: Optional[List[float]] = None
    ) -> List[Dict[str, np.ndarray]]:
        """
        Predict spectra for a batch of peptide sequences.
        
        Args:
            sequences: List of amino acid sequences
            precursor_mz_list: List of precursor m/z values
            charge_list: List of charge states
            max_intensity_list: List of max intensities for denormalization
            
        Returns:
            List of prediction dictionaries
        """
        if max_intensity_list is None:
            max_intensity_list = [1.0] * len(sequences)
        
        batch_size = len(sequences)
        
        # Tokenize sequences
        tokens_list = [
            self.tokenizer.encode(seq, max_length=self.model.max_length, padding=True)
            for seq in sequences
        ]
        tokens_tensor = torch.LongTensor(tokens_list).to(self.device)
        
        # Create masks
        mask = tokens_tensor != self.tokenizer.pad_idx
        
        # Prepare metadata
        norm_precursor_mz = torch.tensor(
            [pm / self.max_mz for pm in precursor_mz_list],
            dtype=torch.float32
        ).to(self.device)
        
        charge_tensor = torch.tensor(charge_list, dtype=torch.long).to(self.device)
        
        # Predict
        pred_mz, pred_intensity, pred_confidence = self.model(
            tokens_tensor,
            mask,
            norm_precursor_mz,
            charge_tensor
        )
        
        # Move to CPU
        pred_mz = pred_mz.cpu().numpy()
        pred_intensity = pred_intensity.cpu().numpy()
        pred_confidence = pred_confidence.cpu().numpy()
        
        # Denormalize and filter for each sample
        results = []
        for i in range(batch_size):
            # Denormalize m/z
            mz = pred_mz[i] * self.max_mz
            
            # Denormalize intensity
            intensity = pred_intensity[i] * max_intensity_list[i]
            
            # Filter by confidence
            confident_mask = pred_confidence[i] > self.confidence_threshold
            
            results.append({
                'mz': mz[confident_mask],
                'intensity': intensity[confident_mask],
                'confidence': pred_confidence[i][confident_mask]
            })
        
        return results
    
    def predict_to_mgf_format(
        self,
        sequence: str,
        precursor_mz: float,
        charge: int,
        max_intensity: float = 1.0
    ) -> str:
        """
        Predict spectrum and format as MGF entry.
        
        Args:
            sequence: Amino acid sequence
            precursor_mz: Precursor m/z value
            charge: Charge state
            max_intensity: Maximum intensity
            
        Returns:
            MGF format string
        """
        prediction = self.predict_single(sequence, precursor_mz, charge, max_intensity)
        
        mgf_lines = [
            "BEGIN IONS",
            f"TITLE=Predicted spectrum for {sequence}",
            f"PEPMASS={precursor_mz}",
            f"CHARGE={charge}+",
            f"SEQ={sequence}"
        ]
        
        # Add peaks
        for mz, intensity in zip(prediction['mz'], prediction['intensity']):
            mgf_lines.append(f"{mz:.4f} {intensity:.2f}")
        
        mgf_lines.append("END IONS")
        
        return '\n'.join(mgf_lines)
    
    def evaluate_prediction(
        self,
        prediction: Dict[str, np.ndarray],
        target_mz: np.ndarray,
        target_intensity: np.ndarray,
        tolerance: float = 0.1
    ) -> Dict[str, float]:
        """
        Evaluate prediction quality against target spectrum.
        
        Args:
            prediction: Predicted spectrum
            target_mz: Target m/z values
            target_intensity: Target intensity values
            tolerance: m/z tolerance for matching (in Da)
            
        Returns:
            Dictionary of evaluation metrics
        """
        pred_mz = prediction['mz']
        pred_intensity = prediction['intensity']
        
        # Count matched peaks
        num_matched = 0
        for target_m in target_mz:
            if np.any(np.abs(pred_mz - target_m) < tolerance):
                num_matched += 1
        
        precision = num_matched / len(pred_mz) if len(pred_mz) > 0 else 0.0
        recall = num_matched / len(target_mz) if len(target_mz) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'num_predicted': len(pred_mz),
            'num_target': len(target_mz),
            'num_matched': num_matched
        }


def load_model_for_inference(
    checkpoint_path: str,
    device: Optional[torch.device] = None
) -> Tuple[MSPredictor, Dict]:
    """
    Load trained model from checkpoint for inference.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model to
        
    Returns:
        Tuple of (model, config_dict)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract config
    config = checkpoint['config']
    
    # Create model
    model = MSPredictor(
        vocab_size=config.model.vocab_size,
        hidden_dim=config.model.hidden_dim,
        num_encoder_layers=config.model.num_encoder_layers,
        num_decoder_layers=config.model.num_decoder_layers,
        num_heads=config.model.num_heads,
        dim_feedforward=config.model.dim_feedforward,
        num_predictions=config.model.num_predictions,
        max_length=config.model.max_length,
        max_charge=config.model.max_charge,
        dropout=config.model.dropout,
        activation=config.model.activation
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, config

