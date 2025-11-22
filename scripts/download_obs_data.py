#!/usr/bin/env python
"""
Download MS/MS data from OBS bucket using obsutil (Parquet format).

Data sources:
- MassIVE-KB (Parquet): obs://lingtz/ultraprot/data-1/MassIVE_KB/
- ProteomeTools: obs://lingtz/ultraprot/data-1/ (other subdirectories)
"""

import argparse
import os
import subprocess
import pandas as pd
import numpy as np
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
        bucket_path: Path in bucket (e.g., obs://lingtz/ultraprot/data-1/MassIVE_KB/)
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
            # Extract just the path part (before size info)
            parts = line.split()
            if parts:
                path = parts[0]
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


def download_massive_kb_parquet(
    output_dir: str,
    num_files: int = None,
    parts: list = None
) -> str:
    """
    Download MassIVE-KB Parquet data from OBS.
    
    Args:
        output_dir: Local output directory
        num_files: Number of files to download (None for all)
        parts: List of part numbers to download (e.g., [1, 5, 25])
               If None, downloads all available parts
        
    Returns:
        Path to downloaded data directory
    """
    print(f"\n=== Downloading MassIVE-KB Parquet Data ===\n")
    
    bucket_path = "obs://lingtz/ultraprot/data-1/MassIVE_KB/"
    local_dir = os.path.join(output_dir, "MassIVE_KB_parquet")
    
    # List Parquet files in bucket
    print("Fetching file list from OBS...")
    all_files = list_bucket_files(bucket_path, ".parquet")
    
    parquet_files = [f for f in all_files if f.endswith('.parquet')]
    
    print(f"Found {len(parquet_files)} Parquet files")
    
    # Filter by part numbers if specified
    if parts:
        filtered_files = []
        for part_num in parts:
            part_pattern = f"part_{part_num:04d}.parquet"
            matching = [f for f in parquet_files if part_pattern in f]
            filtered_files.extend(matching)
        parquet_files = filtered_files
        print(f"Filtered to {len(parquet_files)} files based on requested parts: {parts}")
    
    # Limit number of files if specified
    if num_files:
        parquet_files = parquet_files[:num_files]
        print(f"Downloading first {num_files} files...")
    
    # Download files
    downloaded_files = []
    total_size = 0
    
    for obs_path in tqdm(parquet_files, desc="Downloading"):
        filename = os.path.basename(obs_path)
        local_path = os.path.join(local_dir, filename)
        
        if download_file(obs_path, local_path):
            downloaded_files.append(local_path)
            if os.path.exists(local_path):
                total_size += os.path.getsize(local_path)
    
    print(f"\nDownloaded {len(downloaded_files)} files to {local_dir}")
    print(f"Total size: {total_size / (1024**3):.2f} GB")
    
    return local_dir


def inspect_parquet_file(parquet_path: str, num_rows: int = 5):
    """
    Inspect the structure of a Parquet file.
    
    Args:
        parquet_path: Path to Parquet file
        num_rows: Number of rows to display
    """
    print(f"\nInspecting Parquet file: {parquet_path}")
    print("=" * 80)
    
    try:
        # Read parquet file
        df = pd.read_parquet(parquet_path)
        
        print(f"\nShape: {df.shape} (rows, columns)")
        print(f"\nColumns: {list(df.columns)}")
        print(f"\nData types:")
        print(df.dtypes)
        
        print(f"\nFirst {num_rows} rows:")
        print(df.head(num_rows))
        
        # Check for specific columns
        expected_cols = ['sequence', 'precursor_mz', 'charge', 'mz', 'intensity']
        missing_cols = [col for col in expected_cols if col not in df.columns]
        if missing_cols:
            print(f"\n⚠ Warning: Missing expected columns: {missing_cols}")
            print(f"Available columns: {list(df.columns)}")
        
        # Show example spectrum data
        if 'mz' in df.columns and 'intensity' in df.columns:
            print(f"\nExample spectrum (first row):")
            print(f"  m/z values: {df['mz'].iloc[0][:10]}... (showing first 10)")
            print(f"  Intensities: {df['intensity'].iloc[0][:10]}... (showing first 10)")
            print(f"  Number of peaks: {len(df['mz'].iloc[0])}")
        
        return df
    
    except Exception as e:
        print(f"Error reading Parquet file: {e}")
        return None


def create_metadata_file(data_dir: str, output_file: str):
    """
    Create a metadata JSON file listing all downloaded Parquet files.
    
    Args:
        data_dir: Directory containing Parquet files
        output_file: Output JSON file path
    """
    print(f"\nCreating metadata file...")
    
    parquet_files = []
    total_spectra = 0
    
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.parquet'):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, data_dir)
                
                # Get file size
                size_mb = os.path.getsize(full_path) / (1024 * 1024)
                
                # Try to read and count rows
                try:
                    df = pd.read_parquet(full_path)
                    num_spectra = len(df)
                    total_spectra += num_spectra
                except Exception as e:
                    num_spectra = -1
                    print(f"Warning: Could not read {file}: {e}")
                
                parquet_files.append({
                    'filename': file,
                    'relative_path': rel_path,
                    'absolute_path': full_path,
                    'size_mb': round(size_mb, 2),
                    'num_spectra': num_spectra
                })
    
    metadata = {
        'data_dir': data_dir,
        'num_files': len(parquet_files),
        'total_size_mb': sum(f['size_mb'] for f in parquet_files),
        'total_spectra': total_spectra,
        'files': parquet_files
    }
    
    with open(output_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved to {output_file}")
    print(f"Total files: {len(parquet_files)}")
    print(f"Total size: {metadata['total_size_mb']:.2f} MB ({metadata['total_size_mb']/1024:.2f} GB)")
    print(f"Total spectra: {total_spectra:,}")


def main():
    parser = argparse.ArgumentParser(
        description='Download MS/MS Parquet data from OBS bucket',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download 5 Parquet files to default location (data/)
  python scripts/download_obs_data.py --num_files 5

  # Download specific parts to custom location
  python scripts/download_obs_data.py --parts 1 5 25 --output_dir /mnt/data/ms

  # Download and inspect
  python scripts/download_obs_data.py --num_files 1 --inspect_first

  # Just inspect an existing file
  python scripts/download_obs_data.py --inspect mydata/part_0001.parquet
        """
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data',
        help='Output directory for downloaded data (default: data/)'
    )
    parser.add_argument(
        '--num_files',
        type=int,
        default=None,
        help='Number of files to download (None for all)'
    )
    parser.add_argument(
        '--parts',
        type=int,
        nargs='+',
        default=None,
        help='Specific part numbers to download (e.g., --parts 1 5 25)'
    )
    parser.add_argument(
        '--inspect',
        type=str,
        default=None,
        help='Inspect a specific Parquet file'
    )
    parser.add_argument(
        '--inspect_first',
        action='store_true',
        help='Inspect the first downloaded file'
    )
    parser.add_argument(
        '--create_metadata',
        action='store_true',
        help='Create metadata JSON file after download'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force redownload even if files exist'
    )
    
    args = parser.parse_args()
    
    # Check if obsutil exists
    if not os.path.exists(OBSUTIL_PATH):
        print(f"Error: obsutil not found at {OBSUTIL_PATH}")
        print("Please configure obsutil first")
        return 1
    
    # If inspect mode, just inspect the file
    if args.inspect:
        inspect_parquet_file(args.inspect)
        return 0
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {os.path.abspath(args.output_dir)}")
    
    # Download data
    data_dir = download_massive_kb_parquet(
        args.output_dir,
        num_files=args.num_files,
        parts=args.parts
    )
    
    # Inspect first file if requested
    if args.inspect_first:
        parquet_files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
        if parquet_files:
            first_file = os.path.join(data_dir, parquet_files[0])
            inspect_parquet_file(first_file)
    
    # Create metadata
    if args.create_metadata:
        metadata_file = os.path.join(args.output_dir, 'massive_kb_metadata.json')
        create_metadata_file(data_dir, metadata_file)
    
    print("\n" + "=" * 80)
    print("Download complete!")
    print("=" * 80)
    print(f"\nData saved to: {os.path.abspath(data_dir)}")
    print("\nNext steps:")
    print("1. Inspect a downloaded Parquet file:")
    print(f"   python scripts/download_obs_data.py --inspect {data_dir}/<filename>.parquet")
    print("\n2. Update configuration to use Parquet data:")
    print(f"   data.train_data_path: {os.path.abspath(data_dir)}")
    print("\n3. Train the model:")
    print("   python scripts/train.py --config configs/obs_data_config.yaml")
    
    return 0


if __name__ == '__main__':
    exit(main())
