#!/usr/bin/env python
"""
Quick test script to verify OBS connection and data download.
"""

import os
import subprocess
import sys

OBSUTIL_PATH = "/root/ms/obsutil_linux_amd64_5.7.9/obsutil"


def test_obsutil():
    """Test if obsutil is configured correctly."""
    print("=" * 60)
    print("Testing OBS Connection")
    print("=" * 60)
    
    if not os.path.exists(OBSUTIL_PATH):
        print(f"✗ obsutil not found at {OBSUTIL_PATH}")
        return False
    
    print(f"✓ obsutil found at {OBSUTIL_PATH}")
    
    # Test listing bucket
    print("\nTesting bucket access...")
    cmd = [OBSUTIL_PATH, "ls", "obs://lingtz/ultraprot/", "-limit", "5"]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"✗ Failed to access bucket")
        print(f"Error: {result.stderr}")
        return False
    
    print("✓ Successfully accessed obs://lingtz/ultraprot/")
    
    # Show first few entries
    lines = result.stdout.split('\n')
    print("\nFirst few entries:")
    for line in lines[:10]:
        if line.strip() and line.strip().startswith('obs://'):
            print(f"  {line.strip()}")
    
    return True


def test_download_single_file():
    """Test downloading a single file."""
    print("\n" + "=" * 60)
    print("Testing Single File Download")
    print("=" * 60)
    
    # Try to download a small file
    test_file = "obs://lingtz/ultraprot/MassIVE_KB1/MassIVE_KB/part_0/00576_A02_P004283_B0I_A00_R1.mgf.hdf5"
    output_dir = "data/test_download"
    output_file = os.path.join(output_dir, "test_file.hdf5")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Downloading: {test_file}")
    print(f"To: {output_file}")
    
    cmd = [OBSUTIL_PATH, "cp", test_file, output_file, "-f"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"✗ Failed to download file")
        print(f"Error: {result.stderr}")
        return False
    
    if os.path.exists(output_file):
        file_size = os.path.getsize(output_file) / (1024 * 1024)
        print(f"✓ Successfully downloaded file ({file_size:.2f} MB)")
        
        # Try to open with h5py
        try:
            import h5py
            with h5py.File(output_file, 'r') as f:
                print(f"✓ Successfully opened HDF5 file")
                print(f"\nHDF5 keys: {list(f.keys())}")
        except ImportError:
            print("⚠ h5py not installed, cannot verify HDF5 structure")
            print("  Install with: pip install h5py")
        except Exception as e:
            print(f"✗ Failed to open HDF5 file: {e}")
            return False
        
        return True
    
    return False


def main():
    print("\n" + "=" * 60)
    print("OBS Data Setup Test")
    print("=" * 60)
    print()
    
    # Test 1: Check obsutil
    if not test_obsutil():
        print("\n✗ OBS connection test failed")
        print("\nPlease ensure obsutil is properly configured:")
        print("1. Check that obsutil is installed")
        print("2. Run obsutil config to set up credentials")
        return 1
    
    # Test 2: Download a file
    if not test_download_single_file():
        print("\n✗ File download test failed")
        return 1
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    print("\nYou're ready to download data!")
    print("\nNext steps:")
    print("1. Download MassIVE-KB data:")
    print("   python scripts/download_obs_data.py --source massive-kb --num_files 10")
    print()
    print("2. Inspect a downloaded HDF5 file:")
    print("   python scripts/download_obs_data.py --inspect data/MassIVE_KB/part_0/<file>.hdf5")
    print()
    print("3. Train the model:")
    print("   python scripts/train.py --config configs/obs_data_config.yaml")
    
    return 0


if __name__ == '__main__':
    exit(main())

