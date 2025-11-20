"""
Utility functions for MS predictor.
"""

import torch
import numpy as np
from typing import Dict, List


def compute_spectral_angle(
    pred_mz: np.ndarray,
    pred_intensity: np.ndarray,
    target_mz: np.ndarray,
    target_intensity: np.ndarray,
    bin_size: float = 1.0,
    max_mz: float = 2000.0
) -> float:
    """
    Compute spectral angle similarity between predicted and target spectra.
    
    Args:
        pred_mz: Predicted m/z values
        pred_intensity: Predicted intensities
        target_mz: Target m/z values
        target_intensity: Target intensities
        bin_size: Size of m/z bins
        max_mz: Maximum m/z value
        
    Returns:
        Spectral angle (0 to π/2)
    """
    num_bins = int(max_mz / bin_size)
    
    # Create binned spectra
    pred_binned = np.zeros(num_bins)
    target_binned = np.zeros(num_bins)
    
    for mz, intensity in zip(pred_mz, pred_intensity):
        bin_idx = int(mz / bin_size)
        if 0 <= bin_idx < num_bins:
            pred_binned[bin_idx] += intensity
    
    for mz, intensity in zip(target_mz, target_intensity):
        bin_idx = int(mz / bin_size)
        if 0 <= bin_idx < num_bins:
            target_binned[bin_idx] += intensity
    
    # Normalize
    pred_norm = np.linalg.norm(pred_binned)
    target_norm = np.linalg.norm(target_binned)
    
    if pred_norm == 0 or target_norm == 0:
        return np.pi / 2
    
    pred_binned = pred_binned / pred_norm
    target_binned = target_binned / target_norm
    
    # Compute cosine similarity
    cos_sim = np.dot(pred_binned, target_binned)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    
    # Return angle
    return np.arccos(cos_sim)


def compute_peak_overlap(
    pred_mz: np.ndarray,
    target_mz: np.ndarray,
    tolerance: float = 0.1
) -> Dict[str, float]:
    """
    Compute peak overlap metrics.
    
    Args:
        pred_mz: Predicted m/z values
        target_mz: Target m/z values
        tolerance: m/z tolerance for matching
        
    Returns:
        Dictionary of metrics
    """
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
        'num_matched': num_matched
    }


def save_spectrum_to_mgf(
    sequences: List[str],
    precursor_mz_list: List[float],
    charge_list: List[int],
    mz_lists: List[np.ndarray],
    intensity_lists: List[np.ndarray],
    output_path: str
):
    """
    Save spectra to MGF file.
    
    Args:
        sequences: List of peptide sequences
        precursor_mz_list: List of precursor m/z values
        charge_list: List of charge states
        mz_lists: List of m/z arrays
        intensity_lists: List of intensity arrays
        output_path: Output file path
    """
    with open(output_path, 'w') as f:
        for seq, prec_mz, charge, mz, intensity in zip(
            sequences, precursor_mz_list, charge_list, mz_lists, intensity_lists
        ):
            f.write("BEGIN IONS\n")
            f.write(f"SEQ={seq}\n")
            f.write(f"PEPMASS={prec_mz}\n")
            f.write(f"CHARGE={charge}+\n")
            
            for m, i in zip(mz, intensity):
                f.write(f"{m:.4f} {i:.2f}\n")
            
            f.write("END IONS\n\n")


def load_spectrum_from_mgf(mgf_path: str) -> List[Dict]:
    """
    Load spectra from MGF file.
    
    Args:
        mgf_path: Path to MGF file
        
    Returns:
        List of spectrum dictionaries
    """
    spectra = []
    current_spectrum = None
    
    with open(mgf_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            if line == "BEGIN IONS":
                current_spectrum = {
                    'mz': [],
                    'intensity': []
                }
            elif line == "END IONS":
                if current_spectrum:
                    spectra.append(current_spectrum)
                current_spectrum = None
            elif current_spectrum is not None:
                if line.startswith("SEQ="):
                    current_spectrum['sequence'] = line.split('=')[1]
                elif line.startswith("PEPMASS="):
                    current_spectrum['precursor_mz'] = float(line.split('=')[1])
                elif line.startswith("CHARGE="):
                    charge_str = line.split('=')[1].rstrip('+')
                    current_spectrum['charge'] = int(charge_str)
                elif ' ' in line and not line.startswith(("TITLE=", "RTINSECONDS=")):
                    # Peak line
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            mz = float(parts[0])
                            intensity = float(parts[1])
                            current_spectrum['mz'].append(mz)
                            current_spectrum['intensity'].append(intensity)
                        except ValueError:
                            pass
    
    # Convert lists to numpy arrays
    for spectrum in spectra:
        spectrum['mz'] = np.array(spectrum['mz'])
        spectrum['intensity'] = np.array(spectrum['intensity'])
    
    return spectra

