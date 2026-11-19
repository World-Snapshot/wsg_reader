#!/usr/bin/env python3
import os
import glob
import subprocess
from pathlib import Path
import argparse
from tqdm import tqdm

def get_frame_pattern(crop_frame_dir):
    """
    Detect the frame naming pattern in crop_frame directory
    Returns the pattern and frame count
    """
    # Check for different possible patterns
    patterns = [
        ('frame_%08d.png', 'frame_*.png'),
        ('frame_%08d.jpg', 'frame_*.jpg'),
        ('%08d.png', '[0-9]*.png'),
        ('%08d.jpg', '[0-9]*.jpg'),
        ('frame_%06d.png', 'frame_*.png'),
        ('frame_%06d.jpg', 'frame_*.jpg'),
        ('%06d.png', '[0-9]*.png'),
        ('%06d.jpg', '[0-9]*.jpg'),
        ('crop_frame_%06d.png', 'crop_frame_*.png'),
        ('crop_frame_%06d.jpg', 'crop_frame_*.jpg')
    ]
    
    for ffmpeg_pattern, glob_pattern in patterns:
        frames = sorted(glob.glob(os.path.join(crop_frame_dir, glob_pattern)))
        if frames:
            # Check if frames are numbered sequentially
            frame_numbers = []
            for frame in frames:
                basename = os.path.basename(frame)
                # Extract number from filename
                num_str = ''.join(filter(str.isdigit, basename.split('.')[0]))
                if num_str:
                    frame_numbers.append(int(num_str))
            
            # Check if sequential starting from 0 or 1
            if frame_numbers:
                min_num = min(frame_numbers)
                max_num = max(frame_numbers)
                if min_num in [0, 1] and max_num - min_num + 1 == len(frame_numbers):
                    # Adjust pattern for correct starting number
                    if min_num == 1:
                        # FFmpeg by default starts from 1
                        return ffmpeg_pattern, len(frames)
                    else:
                        # Need to specify start number for FFmpeg
                        return ffmpeg_pattern, len(frames), min_num
    
    return None, 0

def create_video_from_frames(video_folder, dry_run=False):
    """
    Create video from crop_frame images for a single video folder
    """
    crop_frame_dir = os.path.join(video_folder, 'crop_frame')
    output_video = os.path.join(video_folder, 'crop_original_video.mp4')
    
    # Check if crop_frame directory exists
    if not os.path.exists(crop_frame_dir):
        return False, "crop_frame directory not found"
    
    # Check if output video already exists
    if os.path.exists(output_video) and not dry_run:
        return False, "crop_original_video.mp4 already exists"
    
    # Get frame pattern and count
    pattern_info = get_frame_pattern(crop_frame_dir)
    if pattern_info[0] is None:
        return False, "No valid frame pattern found"
    
    pattern = pattern_info[0]
    frame_count = pattern_info[1]
    start_number = pattern_info[2] if len(pattern_info) > 2 else 1
    
    if frame_count == 0:
        return False, "No frames found in crop_frame directory"
    
    # Construct FFmpeg command
    # Using -r 1 for 1fps, -crf 23 for good quality, -pix_fmt yuv420p for compatibility
    cmd = [
        'ffmpeg',
        '-y',  # Overwrite output file
        '-framerate', '1',  # Input framerate
        '-start_number', str(start_number),
        '-i', os.path.join(crop_frame_dir, pattern),
        '-c:v', 'libx264',  # H.264 codec
        '-r', '1',  # Output framerate
        '-pix_fmt', 'yuv420p',  # Pixel format for compatibility
        '-crf', '23',  # Quality (lower is better, 23 is default)
        output_video
    ]
    
    if dry_run:
        print(f"Would run: {' '.join(cmd)}")
        return True, f"Would create video with {frame_count} frames"
    
    try:
        # Run FFmpeg command
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return False, f"FFmpeg error: {result.stderr}"
        return True, f"Created video with {frame_count} frames"
    except Exception as e:
        return False, f"Error running FFmpeg: {str(e)}"

def process_dataset(dataset_path, dry_run=False):
    """
    Process all videos in a dataset
    """
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path not found: {dataset_path}")
        return 0, 0
    
    # Get all video folders
    video_folders = [d for d in glob.glob(os.path.join(dataset_path, '*')) 
                     if os.path.isdir(d) and not os.path.basename(d).endswith('.json')]
    
    if not video_folders:
        print(f"No video folders found in {dataset_path}")
        return 0, 0
    
    success_count = 0
    skip_count = 0
    error_count = 0
    
    print(f"\nProcessing {len(video_folders)} videos in {os.path.basename(dataset_path)}...")
    
    for video_folder in tqdm(video_folders, desc=f"Processing {os.path.basename(dataset_path)}"):
        success, message = create_video_from_frames(video_folder, dry_run)
        
        if success:
            success_count += 1
        else:
            if "already exists" in message:
                skip_count += 1
            else:
                error_count += 1
                print(f"\nError in {os.path.basename(video_folder)}: {message}")
    
    return success_count, skip_count, error_count

def main():
    parser = argparse.ArgumentParser(description='Convert crop_frame images to videos for AVIS datasets')
    parser.add_argument('--dataset', choices=['test', 'train', 'val', 'all'], 
                        default='all', help='Which dataset to process')
    parser.add_argument('--dry-run', action='store_true', 
                        help='Show what would be done without actually creating videos')
    parser.add_argument('--base-path', default='/research/cbim/vast/sf895/code/wsg/output',
                        help='Base path for datasets')
    
    args = parser.parse_args()
    
    # Define dataset paths
    datasets = {
        'test': os.path.join(args.base_path, 'AVIS_test_results'),
        'train': os.path.join(args.base_path, 'AVIS_train_results'),
        'val': os.path.join(args.base_path, 'AVIS_val_results')
    }
    
    # Determine which datasets to process
    if args.dataset == 'all':
        datasets_to_process = ['test', 'train', 'val']
    else:
        datasets_to_process = [args.dataset]
    
    # Process each dataset
    total_success = 0
    total_skip = 0
    total_error = 0
    
    for dataset_name in datasets_to_process:
        dataset_path = datasets[dataset_name]
        print(f"\n{'='*60}")
        print(f"Processing AVIS_{dataset_name}_results")
        print(f"Path: {dataset_path}")
        print(f"{'='*60}")
        
        success, skip, error = process_dataset(dataset_path, args.dry_run)
        total_success += success
        total_skip += skip
        total_error += error
        
        print(f"\nDataset {dataset_name} summary:")
        print(f"  - Successfully created: {success}")
        print(f"  - Skipped (already exists): {skip}")
        print(f"  - Errors: {error}")
    
    # Print overall summary
    print(f"\n{'='*60}")
    print(f"Overall Summary - {'DRY RUN' if args.dry_run else 'COMPLETED'}")
    print(f"{'='*60}")
    print(f"Total videos processed: {total_success + total_skip + total_error}")
    print(f"Successfully created: {total_success}")
    print(f"Skipped (already exists): {total_skip}")
    print(f"Errors: {total_error}")
    
    if args.dry_run:
        print("\nThis was a dry run. Use without --dry-run to actually create videos.")

if __name__ == '__main__':
    main()