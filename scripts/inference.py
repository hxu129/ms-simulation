#!/usr/bin/env python
"""
Inference script for MS spectrum predictor.
"""

import argparse
import torch
import numpy as np

import sys
sys.path.insert(0, '/root/ms/src')

from ms_predictor.inference.predictor import SpectrumPredictor, load_model_for_inference
from ms_predictor.data.tokenizer import AminoAcidTokenizer


def main():
    parser = argparse.ArgumentParser(description='Predict MS spectrum from peptide sequence')
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--sequence',
        type=str,
        required=True,
        help='Peptide amino acid sequence'
    )
    parser.add_argument(
        '--precursor_mz',
        type=float,
        required=True,
        help='Precursor m/z value'
    )
    parser.add_argument(
        '--charge',
        type=int,
        required=True,
        help='Charge state'
    )
    parser.add_argument(
        '--confidence_threshold',
        type=float,
        default=0.5,
        help='Confidence threshold for filtering predictions'
    )
    parser.add_argument(
        '--max_intensity',
        type=float,
        default=1000.0,
        help='Maximum intensity for denormalization'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path (MGF format). If not specified, prints to console'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda or cpu)'
    )
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model, config = load_model_for_inference(args.model_path, device)
    print("Model loaded successfully")
    
    # Create predictor
    tokenizer = AminoAcidTokenizer()
    predictor = SpectrumPredictor(
        model=model,
        tokenizer=tokenizer,
        confidence_threshold=args.confidence_threshold,
        max_mz=config.data.max_mz,
        device=device
    )
    
    # Make prediction
    print(f"\nPredicting spectrum for sequence: {args.sequence}")
    print(f"Precursor m/z: {args.precursor_mz}")
    print(f"Charge: {args.charge}")
    
    prediction = predictor.predict_single(
        sequence=args.sequence,
        precursor_mz=args.precursor_mz,
        charge=args.charge,
        max_intensity=args.max_intensity
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
    if args.output:
        mgf_output = predictor.predict_to_mgf_format(
            sequence=args.sequence,
            precursor_mz=args.precursor_mz,
            charge=args.charge,
            max_intensity=args.max_intensity
        )
        
        with open(args.output, 'w') as f:
            f.write(mgf_output)
        
        print(f"\nPrediction saved to {args.output}")


if __name__ == '__main__':
    main()

