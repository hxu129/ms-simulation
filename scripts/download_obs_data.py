#!/usr/bin/env python
"""
Download MS/MS data from OBS bucket using obsutil.

Data sources:
- MassIVE-KB: obs://lingtz/ultraprot/MassIVE_KB1/
- ProteomeTools: obs://lingtz/ultraprot/data-1/
"""

import argparse
import os
import subprocess
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json


OBSUTIL_PATH = "/root/ms/obsutil_linux_amd64_5.7.9/obsutil"


def run_obsutil_command(cmd: list) -> tuple:
    """
    Run obsutil command and return output.
    
    Args:
        cmd: Command as list of strings
        
    Returns:
        Tuple of (returncode, stdout, stderr)
    """
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )
    return result.returncode, result.stdout, result.stderr


def list_bucket_files(bucket_path: str, pattern: str = "") -> list:
    """
    List files in OBS bucket.
    
    Args:
        bucket_path: Path in bucket (e.g., obs://lingtz/ultraprot/MassIVE_KB1/)
        pattern: Filter pattern (optional)
        
    Returns:
        List of file paths
    """
    print(f"Listing files in {bucket_path}...")
    
    cmd = [OBSUTIL_PATH, "ls", bucket_path, "-r"]
    returncode, stdout, stderr = run_obsutil_command(cmd)
    
    if returncode != 0:
        print(f"Error listing files: {stderr}")
        return []
    
    # Parse output to get file paths
    files = []
    for line in stdout.split('\n'):
        line = line.strip()
        if line.startswith('obs://') and (not pattern or pattern in line):
            # Extract just the path part
            if '\t' in line:
                path = line.split('\t')[0].strip()
            else:
                path = line.split()[0] if ' ' in line else line
            files.append(path)
    
    return files


def download_file(obs_path: str, local_path: str, force: bool = False) -> bool:
    """
    Download a single file from OBS.
    
    Args:
        obs_path: OBS file path (obs://...)
        local_path: Local file path
        force: Force redownload if file exists
        
    Returns:
        True if successful, False otherwise
    """
    # Check if file already exists
    if os.path.exists(local_path) and not force:
        return True
    
    # Create directory if needed
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    # Download
    cmd = [OBSUTIL_PATH, "cp", obs_path, local_path, "-f"]
    returncode, stdout, stderr = run_obsutil_command(cmd)
    
    if returncode != 0:
        print(f"Error downloading {obs_path}: {stderr}")
        return False
    
    return True


def download_massive_kb_data(
    output_dir: str,
    num_files: int = None,
    part: int = 0
) -> str:
    """
    Download MassIVE-KB data from OBS.
    
    Args:
        output_dir: Local output directory
        num_files: Number of files to download (None for all)
        part: Which part to download (0, 1, 2, ...)
        
    Returns:
        Path to downloaded data directory
    """
    print(f"\n=== Downloading MassIVE-KB Data (Part {part}) ===\n")
    
    bucket_path = f"obs://lingtz/ultraprot/MassIVE_KB1/MassIVE_KB/part_{part}/"
    local_dir = os.path.join(output_dir, f"MassIVE_KB/part_{part}")
    
    # List HDF5 files in bucket
    print("Fetching file list from OBS...")
    all_files = list_bucket_files(bucket_path, ".hdf5")
    
    # Filter to get only HDF5 files (not MGF)
    hdf5_files = [f for f in all_files if f.endswith('.hdf5') and not f.endswith('.mgf.hdf5')]
    
    if not hdf5_files:
        # If no pure HDF5 files, get MGF.HDF5 files
        hdf5_files = [f for f in all_files if f.endswith('.mgf.hdf5')]
    
    print(f"Found {len(hdf5_files)} HDF5 files")
    
    if num_files:
        hdf5_files = hdf5_files[:num_files]
        print(f"Downloading first {num_files} files...")
    
    # Download files
    downloaded_files = []
    for obs_path in tqdm(hdf5_files, desc="Downloading"):
        filename = os.path.basename(obs_path)
        local_path = os.path.join(local_dir, filename)
        
        if download_file(obs_path, local_path):
            downloaded_files.append(local_path)
    
    print(f"\nDownloaded {len(downloaded_files)} files to {local_dir}")
    
    return local_dir


def download_proteometools_data(
    output_dir: str,
    num_files: int = None
) -> str:
    """
    Download ProteomeTools data from OBS.
    
    Args:
        output_dir: Local output directory
        num_files: Number of files to download (None for all)
        
    Returns:
        Path to downloaded data directory
    """
    print(f"\n=== Downloading ProteomeTools Data ===\n")
    
    bucket_path = "obs://lingtz/ultraprot/data-1/"
    local_dir = os.path.join(output_dir, "ProteomeTools")
    
    # List files in bucket
    print("Fetching file list from OBS...")
    all_files = list_bucket_files(bucket_path, ".hdf5")
    
    hdf5_files = [f for f in all_files if f.endswith('.hdf5')]
    
    print(f"Found {len(hdf5_files)} HDF5 files")
    
    if num_files:
        hdf5_files = hdf5_files[:num_files]
        print(f"Downloading first {num_files} files...")
    
    # Download files
    downloaded_files = []
    for obs_path in tqdm(hdf5_files, desc="Downloading"):
        # Preserve directory structure
        relative_path = obs_path.replace("obs://lingtz/ultraprot/data-1/", "")
        local_path = os.path.join(local_dir, relative_path)
        
        if download_file(obs_path, local_path):
            downloaded_files.append(local_path)
    
    print(f"\nDownloaded {len(downloaded_files)} files to {local_dir}")
    
    return local_dir


def inspect_hdf5_file(hdf5_path: str):
    """
    Inspect the structure of an HDF5 file.
    
    Args:
        hdf5_path: Path to HDF5 file
    """
    print(f"\nInspecting HDF5 file: {hdf5_path}")
    print("=" * 60)
    
    with h5py.File(hdf5_path, 'r') as f:
        def print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"Dataset: {name}")
                print(f"  Shape: {obj.shape}")
                print(f"  Dtype: {obj.dtype}")
                if obj.size < 10:
                    print(f"  Data: {obj[...]}")
                print()
        
        print("HDF5 Structure:")
        print("-" * 60)
        f.visititems(print_structure)
        
        # Print top-level keys
        print("\nTop-level keys:")
        print(list(f.keys()))


def create_metadata_file(data_dir: str, output_file: str):
    """
    Create a metadata JSON file listing all downloaded HDF5 files.
    
    Args:
        data_dir: Directory containing HDF5 files
        output_file: Output JSON file path
    """
    print(f"\nCreating metadata file...")
    
    hdf5_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.hdf5'):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, data_dir)
                
                # Get file size
                size_mb = os.path.getsize(full_path) / (1024 * 1024)
                
                hdf5_files.append({
                    'filename': file,
                    'relative_path': rel_path,
                    'absolute_path': full_path,
                    'size_mb': round(size_mb, 2)
                })
    
    metadata = {
        'data_dir': data_dir,
        'num_files': len(hdf5_files),
        'total_size_mb': sum(f['size_mb'] for f in hdf5_files),
        'files': hdf5_files
    }
    
    with open(output_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved to {output_file}")
    print(f"Total files: {len(hdf5_files)}")
    print(f"Total size: {metadata['total_size_mb']:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description='Download MS/MS data from OBS bucket'
    )
    parser.add_argument(
        '--source',
        type=str,
        choices=['massive-kb', 'proteometools', 'both'],
        default='massive-kb',
        help='Data source to download'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data',
        help='Output directory for downloaded data'
    )
    parser.add_argument(
        '--num_files',
        type=int,
        default=None,
        help='Number of files to download (None for all)'
    )
    parser.add_argument(
        '--part',
        type=int,
        default=0,
        help='Which part to download for MassIVE-KB (0, 1, 2, ...)'
    )
    parser.add_argument(
        '--inspect',
        type=str,
        default=None,
        help='Inspect a specific HDF5 file'
    )
    parser.add_argument(
        '--create_metadata',
        action='store_true',
        help='Create metadata JSON file after download'
    )
    
    args = parser.parse_args()
    
    # Check if obsutil exists
    if not os.path.exists(OBSUTIL_PATH):
        print(f"Error: obsutil not found at {OBSUTIL_PATH}")
        print("Please configure obsutil first")
        return 1
    
    # If inspect mode, just inspect the file
    if args.inspect:
        inspect_hdf5_file(args.inspect)
        return 0
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Download data
    if args.source == 'massive-kb' or args.source == 'both':
        data_dir = download_massive_kb_data(
            args.output_dir,
            num_files=args.num_files,
            part=args.part
        )
        
        if args.create_metadata:
            metadata_file = os.path.join(args.output_dir, 'massive_kb_metadata.json')
            create_metadata_file(data_dir, metadata_file)
    
    if args.source == 'proteometools' or args.source == 'both':
        data_dir = download_proteometools_data(
            args.output_dir,
            num_files=args.num_files
        )
        
        if args.create_metadata:
            metadata_file = os.path.join(args.output_dir, 'proteometools_metadata.json')
            create_metadata_file(data_dir, metadata_file)
    
    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)
    print(f"\nData saved to: {args.output_dir}")
    print("\nNext steps:")
    print("1. Inspect a downloaded HDF5 file:")
    print(f"   python scripts/download_obs_data.py --inspect {args.output_dir}/MassIVE_KB/part_0/<filename>.hdf5")
    print("\n2. Update the HDF5Dataset class in src/ms_predictor/data/dataset.py")
    print("   to parse the HDF5 format")
    print("\n3. Update config to use real data:")
    print("   data.use_dummy_data: false")
    print(f"   data.train_data_path: {args.output_dir}/...")
    
    return 0


if __name__ == '__main__':
    exit(main())

