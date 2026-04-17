"""
Download the entire FFHQ-UV dataset from Hugging Face
Dataset: https://huggingface.co/datasets/csbhr/FFHQ-UV

Supports partial download for testing before downloading full dataset
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

def count_downloaded_files(output_dir):
    """Count the number of files already downloaded."""
    if not output_dir.exists():
        return 0
    
    total_files = 0
    for root, dirs, files in os.walk(output_dir):
        total_files += len(files)
    
    return total_files

def download_ffhq_uv_dataset(test_mode=False, full_download=False):
    """
    Download the FFHQ-UV dataset from Hugging Face.
    
    Args:
        test_mode (bool): If True, download only first half for testing
        full_download (bool): If True, download complete dataset
    """
    
    # Set the output directory
    output_dir = Path(__file__).parent / "FFHQ-UV-Dataset"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check existing downloads
    existing_files = count_downloaded_files(output_dir)
    
    if existing_files > 0:
        print(f"\n📊 Already downloaded: {existing_files} files")
        
        if not full_download and not test_mode:
            print("\n🔍 Download options:")
            print("1. Continue with full dataset download")
            print("2. Download remaining data (if interrupted)")
            print("3. Exit\n")
            
            choice = input("Enter your choice (1-3): ").strip()
            
            if choice == "3":
                print("Exiting...")
                return
            elif choice == "2":
                print("Resuming download from where it left off...\n")
            # choice 1 continues to full download
    
    try:
        print(f"Downloading FFHQ-UV dataset to: {output_dir}")
        print("This may take a while depending on your internet connection...")
        print("Dataset size is approximately 500GB+\n")
        
        # Download the entire dataset
        repo_path = snapshot_download(
            repo_id="csbhr/FFHQ-UV",
            repo_type="dataset",
            local_dir=str(output_dir),
            local_dir_use_symlinks=False,  # Use actual files instead of symlinks
            resume_download=True,  # Resume if interrupted
        )
        
        # Count final files
        final_files = count_downloaded_files(output_dir)
        
        print(f"\n✓ Dataset download complete!")
        print(f"📊 Total files downloaded: {final_files}")
        print(f"📂 Location: {repo_path}")
        print(f"\n💾 Check the directory size with:")
        print(f"   Linux/Mac: du -sh {output_dir}")
        print(f"   Windows:   dir /s {output_dir}")
        print(f"\n✅ You can now use this dataset for training!")
        
        # Offer to continue with full download if in test mode
        if test_mode:
            response = input("\n❓ Download complete. Start using dataset for testing? (y/n): ").strip().lower()
            if response == 'y':
                print("✅ Ready to proceed with training using partial dataset!")
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted by user!")
        final_files = count_downloaded_files(output_dir)
        print(f"📊 Files downloaded so far: {final_files}")
        print(f"💡 Run the script again to resume download from where it stopped.\n")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n✗ Error downloading dataset: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure you have sufficient disk space (500GB+)")
        print("2. Check your internet connection")
        print("3. If the download is interrupted, run the script again - it will resume")
        print("4. For authentication, run: huggingface-cli login")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download FFHQ-UV dataset from Hugging Face")
    parser.add_argument("--test", action="store_true", 
                        help="Test mode: download and check data")
    parser.add_argument("--full", action="store_true", 
                        help="Download full dataset")
    parser.add_argument("--resume", action="store_true", 
                        help="Resume incomplete download")
    
    args = parser.parse_args()
    
    download_ffhq_uv_dataset(test_mode=args.test, full_download=args.full or args.resume)
