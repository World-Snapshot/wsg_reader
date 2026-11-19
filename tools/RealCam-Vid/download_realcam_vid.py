#!/usr/bin/env python3
"""
RealCam-Vid Dataset Downloader
Following the official repository instructions
"""

import os
import subprocess
import sys
from pathlib import Path
import json
from huggingface_hub import HfApi, hf_hub_download

def check_auth_and_access():
    """Check if we can access the dataset before starting download"""
    print("Checking dataset access...")
    try:
        # Try to access the dataset info
        api = HfApi()
        dataset_info = api.dataset_info("MuteApo/RealCam-Vid")
        print(f"✓ Dataset found: {dataset_info.id}")
        
        # Try to download a small file to test access
        test_file = hf_hub_download(
            repo_id="MuteApo/RealCam-Vid",
            filename="README.md",
            repo_type="dataset",
            cache_dir=".cache_test"
        )
        print("✓ Test download successful")
        
        # Clean up test
        if os.path.exists(".cache_test"):
            import shutil
            shutil.rmtree(".cache_test")
            
        return True
        
    except Exception as e:
        print(f"✗ Access check failed: {e}")
        print("\nPossible solutions:")
        print("1. Login to HuggingFace: huggingface-cli login")
        print("2. Set HF_TOKEN environment variable")
        print("3. Check if the dataset requires special access permissions")
        return False

def download_realcam_vid(data_root=None, skip_errors=False):
    """
    Download RealCam-Vid dataset using huggingface-cli as recommended in the official repo
    """
    # Use relative path from main directory
    if data_root is None:
        # Get the path relative to where the script is called from
        data_root = Path("/common/users/sf895/wsg/input/RealCam-Vid").resolve()
    
    print("=" * 80)
    print("RealCam-Vid Dataset Downloader")
    print("Dataset contains ~100K video clips with:")
    print("- Dynamic scenes and camera movements")
    print("- Metric-scale camera annotations")
    print("- MonST3R processed data")
    print("=" * 80)
    
    # Check access first
    if not check_auth_and_access():
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            return False
    
    # Create directory if not exists
    Path(data_root).mkdir(parents=True, exist_ok=True)
    
    # Check what's already downloaded
    existing_files = []
    if Path(data_root).exists():
        for f in Path(data_root).rglob('*'):
            if f.is_file():
                existing_files.append(f.name)
        if existing_files:
            print(f"\nFound {len(existing_files)} existing files, will resume download")
    
    # Set environment variables
    env = os.environ.copy()
    
    # Check if user has HF_TOKEN set
    if 'HF_TOKEN' in env:
        print("✓ Using HF_TOKEN for authentication")
    else:
        print("ℹ No HF_TOKEN found. If download fails with 403 error, you may need to:")
        print("  1. Run: huggingface-cli login")
        print("  2. Or set: export HF_TOKEN=your_token_here")
    
    # Use official endpoint since we're in the US
    # Comment out to use default HuggingFace endpoint
    # env['HF_ENDPOINT'] = 'https://hf-mirror.com'
    print("Using official HuggingFace endpoint")
    
    # Download command from the official repo
    download_cmd = [
        'huggingface-cli', 'download',
        '--repo-type', 'dataset',
        '--resume-download',  # This ensures we resume from where we left off
        'MuteApo/RealCam-Vid',
        '--local-dir', str(data_root),
        '--local-dir-use-symlinks', 'False'
    ]
    
    print(f"\nDownloading to: {data_root}")
    print("This will download approximately 227GB of data:")
    print("- DL3DV-10K (~74G)")
    print("- RealEstate10K (~75G)")
    print("- MiraData9K (~78G)")
    print("\nStarting download...\n")
    
    try:
        # Run download command
        result = subprocess.run(download_cmd, env=env, check=True)
        print("\nDownload completed successfully!")
        
        # Unzip files as instructed
        print("\nUnzipping downloaded files...")
        unzip_downloaded_files(data_root)
        
    except subprocess.CalledProcessError as e:
        print(f"\nError during download: {e}")
        
        # Check if it's authentication error
        if "403" in str(e):
            print("\n⚠️  Authentication Error (403 Forbidden)")
            print("This dataset may require authentication. Please try:")
            print("1. Run: huggingface-cli login")
            print("2. Or: export HF_TOKEN=your_huggingface_token")
            print("\nNote: The download may have partially succeeded.")
            print("Run the script again to resume from where it stopped.")
        
        # Still check what was downloaded
        verify_download(data_root)
        return False
    except KeyboardInterrupt:
        print("\nDownload interrupted by user")
        return False
    
    return True

def unzip_downloaded_files(data_root, delete_after_extract=True):
    """
    Unzip all downloaded zip files using the official method with find and unzip
    """
    import subprocess
    from tqdm import tqdm
    
    zip_dir = Path(data_root) / "zip"
    if not zip_dir.exists():
        print("No zip directory found, skipping unzip step")
        return
    
    # Change to data root directory
    os.chdir(data_root)
    
    print("\nStarting extraction using official method...")
    print("This will extract all zip files and may take some time.")
    
    # Use the official command from the repo
    extract_cmd = '''find "zip" -type f -name "*.zip" -exec sh -c '
        for zip_file do
            echo "Extracting $(basename "$zip_file")..."
            unzip -q "$zip_file"
        done
    ' sh {} +'''
    
    try:
        # Run the extraction command
        print("\nExtracting all zip files...")
        result = subprocess.run(extract_cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("\n✓ Extraction completed successfully!")
        else:
            print(f"\n⚠ Extraction completed with warnings: {result.stderr}")
        
        # If requested, delete zip files after extraction
        if delete_after_extract:
            print("\nCleaning up zip files...")
            cleanup_cmd = 'find "zip" -type f -name "*.zip" -delete'
            subprocess.run(cleanup_cmd, shell=True)
            
            # Remove empty directories
            cleanup_dirs_cmd = 'find "zip" -type d -empty -delete'
            subprocess.run(cleanup_dirs_cmd, shell=True)
            
            print("✓ Cleaned up zip files and empty directories")
            
    except Exception as e:
        print(f"\nError during extraction: {e}")
        return False
    
    # Verify extraction
    print("\nVerifying extracted files...")
    for dataset in ['DL3DV-10K', 'RealEstate10K', 'MiraData9K']:
        dataset_path = Path(data_root) / dataset
        if dataset_path.exists():
            # Count files
            mp4_count = len(list(dataset_path.rglob("*.mp4")))
            dir_count = len([d for d in dataset_path.iterdir() if d.is_dir()])
            print(f"  ✓ {dataset}: {mp4_count} videos in {dir_count} directories")
        else:
            print(f"  ⚠ {dataset}: directory not found")
    
    return True

def verify_download(data_root):
    """
    Verify the downloaded dataset with detailed status
    """
    data_path = Path(data_root)
    
    print("\n" + "="*60)
    print("Download Status Report")
    print("="*60)
    
    # Check for metadata files
    metadata_files = {
        "RealCam-Vid_train.npz": "Training metadata",
        "RealCam-Vid_test.npz": "Test metadata",
        "RealCam-Vid_train.csv": "Training CSV",
        "RealCam-Vid_test.csv": "Test CSV"
    }
    
    print("\nMetadata files:")
    metadata_ok = True
    for filename, desc in metadata_files.items():
        file_path = data_path / filename
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"  ✓ {filename} ({size_mb:.1f} MB) - {desc}")
        else:
            print(f"  ✗ {filename} - {desc}")
            metadata_ok = False
    
    # Check zip files
    print("\nZip files:")
    zip_dir = data_path / "zip"
    total_zips = 0
    total_size_gb = 0
    
    # Check if data has been extracted already
    data_extracted = False
    for dataset in ['DL3DV-10K', 'RealEstate10K', 'MiraData9K']:
        dataset_path = data_path / dataset
        if dataset_path.exists() and list(dataset_path.rglob("*.mp4")):
            data_extracted = True
            break
    
    if zip_dir.exists():
        for dataset in ['DL3DV-10K', 'RealEstate10K', 'MiraData9K']:
            dataset_dir = zip_dir / dataset
            if dataset_dir.exists():
                zip_files = list(dataset_dir.glob("*.zip"))
                dataset_size = sum(f.stat().st_size for f in zip_files) / (1024**3)
                total_zips += len(zip_files)
                total_size_gb += dataset_size
                print(f"  ✓ {dataset}: {len(zip_files)} files ({dataset_size:.1f} GB)")
            else:
                if data_extracted:
                    print(f"  ℹ {dataset}: zip files removed after extraction")
                else:
                    print(f"  ✗ {dataset}: not found")
    else:
        if data_extracted:
            print("  ℹ Zip files have been removed after successful extraction")
        else:
            print("  ✗ No zip directory found")
    
    # Check for extracted directories
    print("\nExtracted data:")
    expected_dirs = ['DL3DV-10K', 'RealEstate10K', 'MiraData9K']
    extracted_ok = True
    total_videos = 0
    for dir_name in expected_dirs:
        dir_path = data_path / dir_name
        if dir_path.exists():
            # Count video files (mp4)
            video_files = list(dir_path.rglob("*.mp4"))
            num_videos = len(video_files)
            total_videos += num_videos
            if num_videos > 0:
                print(f"  ✓ {dir_name}: {num_videos} videos")
            else:
                # Check if there are subdirectories
                subdirs = [d for d in dir_path.iterdir() if d.is_dir()]
                if subdirs:
                    print(f"  ✓ {dir_name}: {len(subdirs)} directories")
                else:
                    print(f"  ⚠ {dir_name}: exists but no videos found")
        else:
            print(f"  ✗ {dir_name}: not extracted yet")
            extracted_ok = False
    
    # Summary
    print("\n" + "-"*60)
    print("Summary:")
    if total_videos > 0:
        print(f"  Total videos: {total_videos:,}")
    print(f"  Total zip files: {total_zips}")
    print(f"  Total size: {total_size_gb:.1f} GB")
    print(f"  Metadata complete: {'Yes' if metadata_ok else 'No'}")
    print(f"  Data extracted: {'Yes' if extracted_ok else 'No'}")
    
    if not metadata_ok:
        print("\n⚠️  Metadata files missing. Please run the download script again.")
    elif not extracted_ok and total_zips == 0 and not data_extracted:
        print("\n⚠️  No data found. Please run the download script to download the dataset.")
    elif not extracted_ok and total_zips > 0:
        print("\n✓ Download complete! Run with --extract flag to extract the data.")
    elif extracted_ok and total_videos > 0:
        print(f"\n✓ Dataset is ready to use! {total_videos:,} videos available.")
    else:
        print("\n✓ Dataset is ready to use!")
    
    return metadata_ok and (total_zips > 0 or extracted_ok)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download RealCam-Vid dataset")
    parser.add_argument("--data-root", type=str, 
                        default=None,
                        help="Root directory for dataset download (default: ../../input/RealCam-Vid)")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only verify existing download")
    parser.add_argument("--extract", action="store_true",
                        help="Extract all zip files after download")
    parser.add_argument("--keep-zips", action="store_true",
                        help="Keep zip files after extraction (default: delete)")
    
    args = parser.parse_args()
    
    # Set default data root if not provided
    if args.data_root is None:
        args.data_root = Path("/common/users/sf895/wsg/input/RealCam-Vid").resolve()
    
    if args.verify_only:
        verify_download(args.data_root)
    elif args.extract:
        # Just extract existing files
        print("Starting extraction of downloaded files...")
        unzip_downloaded_files(args.data_root, delete_after_extract=not args.keep_zips)
        verify_download(args.data_root)
    else:
        # Download and optionally extract
        if download_realcam_vid(args.data_root):
            verify_download(args.data_root)
            
            # Ask user if they want to extract now
            if not args.extract:
                response = input("\nWould you like to extract the zip files now? (y/n): ")
                if response.lower() == 'y':
                    args.extract = True
            
            if args.extract:
                print("\nStarting extraction...")
                unzip_downloaded_files(args.data_root, delete_after_extract=not args.keep_zips)
                verify_download(args.data_root)