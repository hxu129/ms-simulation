#!/usr/bin/env python
"""
Inference script for MS spectrum predictor using Hydra configuration.

Usage examples:
  # Predict with all required parameters via command line
  python inference.py model_path=checkpoints/best_model.pt \
                      sequence=PEPTIDE \
                      precursor_mz=500.5 \
                      charge=2
  
  # With optional parameters
  python inference.py model_path=checkpoints/best_model.pt \
                      sequence=PEPTIDE \
                      precursor_mz=500.5 \
                      charge=2 \
                      inference.confidence_threshold=0.7 \
                      output.file=prediction.mgf
  
  # Using a config file
  python inference.py --config-name inference \
                      model_path=checkpoints/best_model.pt \
                      sequence=PEPTIDE \
                      precursor_mz=500.5 \
                      charge=2
"""

import torch
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf

import sys
sys.path.insert(0, '/root/ms/src')

from ms_predictor.inference.predictor import SpectrumPredictor, load_model_for_inference
from ms_predictor.data.tokenizer import AminoAcidTokenizer


@hydra.main(version_base=None, config_path="../configs", config_name="inference")
def main(cfg: DictConfig):
    """
    Inference function with Hydra configuration.
    
    Args:
        cfg: Hydra configuration object (DictConfig)
    """
    # Print configuration
    print("=" * 80)
    print("Inference Configuration:")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)
    
    # Get the original working directory (Hydra changes cwd to outputs/)
    original_cwd = hydra.utils.get_original_cwd()
    
    # Validate required parameters
    if cfg.model_path == "???":
        raise ValueError("model_path is required. Provide it via command line: model_path=path/to/checkpoint.pt")
    if cfg.sequence == "???":
        raise ValueError("sequence is required. Provide it via command line: sequence=PEPTIDE")
    if cfg.precursor_mz == "???":
        raise ValueError("precursor_mz is required. Provide it via command line: precursor_mz=500.5")
    if cfg.charge == "???":
        raise ValueError("charge is required. Provide it via command line: charge=2")
    
    # Handle relative paths from original working directory
    import os
    model_path = cfg.model_path
    if not os.path.isabs(model_path):
        model_path = os.path.join(original_cwd, model_path)
    
    # Set device
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {model_path}...")
    model, config = load_model_for_inference(model_path, device)
    print("Model loaded successfully")
    
    # Create predictor
    tokenizer = AminoAcidTokenizer()
    predictor = SpectrumPredictor(
        model=model,
        tokenizer=tokenizer,
        confidence_threshold=cfg.inference.confidence_threshold,
        max_mz=cfg.inference.max_mz,
        device=device
    )
    
    # Make prediction
    print(f"\nPredicting spectrum for sequence: {cfg.sequence}")
    print(f"Precursor m/z: {cfg.precursor_mz}")
    print(f"Charge: {cfg.charge}")
    
    prediction = predictor.predict_single(
        sequence=cfg.sequence,
        precursor_mz=float(cfg.precursor_mz),
        charge=int(cfg.charge),
        max_intensity=cfg.inference.max_intensity
    )
    
    # Print results
    print(f"\nPredicted {len(prediction['mz'])} peaks:")
    print(f"{'m/z':<12} {'Intensity':<12} {'Confidence':<12}")
    print("-" * 36)
    
    # Sort by m/z
    sort_idx = np.argsort(prediction['mz'])
    for idx in sort_idx[:20]:  # Print first 20 peaks
        print(f"{prediction['mz'][idx]:<12.4f} "
              f"{prediction['intensity'][idx]:<12.2f} "
              f"{prediction['confidence'][idx]:<12.4f}")
    
    if len(prediction['mz']) > 20:
        print(f"... and {len(prediction['mz']) - 20} more peaks")
    
    # Save to file if specified
    if cfg.output.file is not None:
        mgf_output = predictor.predict_to_mgf_format(
            sequence=cfg.sequence,
            precursor_mz=float(cfg.precursor_mz),
            charge=int(cfg.charge),
            max_intensity=cfg.inference.max_intensity
        )
        
        output_path = cfg.output.file
        if not os.path.isabs(output_path):
            output_path = os.path.join(original_cwd, output_path)
        
        with open(output_path, 'w') as f:
            f.write(mgf_output)
        
        print(f"\nPrediction saved to {output_path}")


if __name__ == '__main__':
    main()
