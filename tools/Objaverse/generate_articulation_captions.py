#!/usr/bin/env python3
"""
Generate frame captions for Articulation_results dataset using Gemini API
Processes three sub-datasets: articulation_xlv2_diverse_pose_test, articulation_xlv2_test, etc.
"""
import os
import csv
import time
import glob
from pathlib import Path
import argparse
from tqdm import tqdm
import google.generativeai as genai
from PIL import Image
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

class APIPool:
    """Pool of Gemini API keys for concurrent processing"""
    def __init__(self, api_keys):
        self.api_keys = api_keys if isinstance(api_keys, list) else [api_keys]
        self.models = []
        self.lock = threading.Lock()
        self.current_index = 0
        
        # Initialize models for each API key
        for api_key in self.api_keys:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.5-flash')
            self.models.append((api_key, model))
    
    def get_model(self):
        """Get next available model in round-robin fashion"""
        with self.lock:
            model_info = self.models[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.models)
            return model_info[1]  # Return the model
    
    def size(self):
        """Return number of available API keys"""
        return len(self.api_keys)

def generate_caption_for_frames(model, frame_paths, video_name):
    """
    Generate caption for single frame or multiple frames
    """
    try:
        # Load images
        images = []
        for frame_path in frame_paths:
            img = Image.open(frame_path)
            images.append(img)
        
        # Prepare prompt based on number of frames
        if len(frame_paths) == 1:
            prompt = "Please provide a detailed description of this 3D rendered person in 2-3 sentences. Focus on the person's pose, articulation, body position, and any visible movements or gestures."
        else:
            prompt = f"These are {len(frame_paths)} sequential frames from a 3D animation. Please provide a detailed description in 2-3 sentences about the person's movement and articulation across these frames. Focus on body poses, joint movements, and the overall motion pattern."
        
        # Generate content
        content = images + [prompt]
        response = model.generate_content(content)
        
        return response.text.strip()
    except Exception as e:
        print(f"Error processing frames for {video_name}: {e}")
        return None

def process_video_folder(model, video_dir, num_frames=3, max_retries=3):
    """
    Process a single video folder and generate caption
    """
    video_name = os.path.basename(video_dir)
    crop_frame_dir = os.path.join(video_dir, 'crop_frame')
    
    if not os.path.exists(crop_frame_dir):
        print(f"No crop_frame directory found for {video_name}")
        return "No crop_frame directory found"
    
    # Get all frame files
    frame_files = sorted([f for f in os.listdir(crop_frame_dir) if f.endswith('.png')])
    
    if not frame_files:
        print(f"No frames found in {video_name}")
        return "No frames found"
    
    # Select frames to process
    if num_frames == 1:
        # Select middle frame
        selected_frames = [frame_files[len(frame_files) // 2]]
    elif len(frame_files) >= num_frames:
        # Select evenly spaced frames
        indices = [int(i * (len(frame_files) - 1) / (num_frames - 1)) for i in range(num_frames)]
        selected_frames = [frame_files[i] for i in indices]
    else:
        # Use all available frames
        selected_frames = frame_files[:num_frames]
    
    frame_paths = [os.path.join(crop_frame_dir, f) for f in selected_frames]
    
    # Try to generate caption with retries
    for attempt in range(max_retries):
        try:
            caption = generate_caption_for_frames(model, frame_paths, video_name)
            if caption:
                return caption
                
        except Exception as e:
            if "quota" in str(e).lower():
                print(f"\nQuota exceeded. Waiting 60 seconds...")
                time.sleep(60)
            elif attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5
                print(f"\nError: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"\nFailed after {max_retries} attempts: {e}")
    
    return "Failed to generate caption"

def get_all_video_folders(articulation_results_path):
    """
    Get all video folders from the three sub-datasets
    """
    sub_datasets = [
        'articulation_xlv2_diverse_pose_test',
        'articulation_xlv2_test',
        'articulation_xlv2_train'  # Add if exists
    ]
    
    video_folders = []
    
    for sub_dataset in sub_datasets:
        sub_dataset_path = os.path.join(articulation_results_path, sub_dataset)
        if not os.path.exists(sub_dataset_path):
            print(f"Sub-dataset not found: {sub_dataset}")
            continue
            
        # Find all video folders in this sub-dataset
        for item in os.listdir(sub_dataset_path):
            item_path = os.path.join(sub_dataset_path, item)
            if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, 'crop_frame')):
                video_folders.append({
                    'path': item_path,
                    'video_name': item,
                    'sub_dataset': sub_dataset
                })
    
    return video_folders

def process_single_video(api_pool, video_info, num_frames=3, max_retries=3):
    """Process a single video using the API pool"""
    model = api_pool.get_model()
    video_path = video_info['path']
    video_name = video_info['video_name']
    sub_dataset = video_info['sub_dataset']
    
    print(f"\nProcessing: {sub_dataset}/{video_name}")
    caption = process_video_folder(model, video_path, num_frames, max_retries)
    
    return {
        'sub_dataset': sub_dataset,
        'video_name': video_name,
        'caption': caption
    }

def process_articulation_dataset(articulation_results_path, api_pool, output_csv, num_frames=3, 
                               skip_existing=True, max_videos=None, max_workers=None):
    """
    Process all videos in Articulation_results dataset and save captions to CSV
    """
    # Get all video folders
    video_folders = get_all_video_folders(articulation_results_path)
    
    if not video_folders:
        print(f"No video folders found in {articulation_results_path}")
        return
    
    print(f"\nFound {len(video_folders)} video folders across all sub-datasets")
    for sub_dataset in set(vf['sub_dataset'] for vf in video_folders):
        count = len([vf for vf in video_folders if vf['sub_dataset'] == sub_dataset])
        print(f"  - {sub_dataset}: {count} videos")
    
    # Limit number of videos if specified
    if max_videos:
        video_folders = video_folders[:max_videos]
        print(f"\nProcessing only first {max_videos} videos")
    
    # Check if CSV already exists and load existing captions
    existing_captions = {}
    if skip_existing and os.path.exists(output_csv):
        with open(output_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = f"{row['sub_dataset']}/{row['video_name']}"
                existing_captions[key] = row['caption']
        print(f"\nFound {len(existing_captions)} existing captions")
    
    # Prepare list of videos to process
    videos_to_process = []
    results = []
    skipped = 0
    
    for video_info in video_folders:
        key = f"{video_info['sub_dataset']}/{video_info['video_name']}"
        
        # Skip if already processed
        if skip_existing and key in existing_captions:
            results.append({
                'sub_dataset': video_info['sub_dataset'],
                'video_name': video_info['video_name'],
                'caption': existing_captions[key]
            })
            skipped += 1
        else:
            videos_to_process.append(video_info)
    
    print(f"\nVideos to process: {len(videos_to_process)}")
    
    # Determine number of workers
    if max_workers is None:
        max_workers = min(api_pool.size() * 2, 10)  # 2 threads per API key, max 10
    
    # Process videos concurrently
    if len(videos_to_process) > 0:
        print(f"Using {max_workers} workers with {api_pool.size()} API keys")
        
        # Create a thread pool
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_video = {
                executor.submit(process_single_video, api_pool, video_info, num_frames): video_info
                for video_info in videos_to_process
            }
            
            # Process completed tasks with progress bar
            with tqdm(total=len(videos_to_process), desc="Processing frames") as pbar:
                completed = 0
                
                for future in as_completed(future_to_video):
                    video_info = future_to_video[future]
                    try:
                        result = future.result()
                        results.append(result)
                        completed += 1
                        pbar.update(1)
                        
                        # Save intermediate results every 20 videos
                        if completed % 20 == 0:
                            save_results_to_csv(results, output_csv)
                            
                    except Exception as e:
                        print(f"\nError processing {video_info['sub_dataset']}/{video_info['video_name']}: {e}")
                        results.append({
                            'sub_dataset': video_info['sub_dataset'],
                            'video_name': video_info['video_name'],
                            'caption': f"Error: {str(e)}"
                        })
                        pbar.update(1)
    
    # Save final results
    save_results_to_csv(results, output_csv)
    print(f"\nCompleted! Processed {len(results) - skipped} new videos, skipped {skipped} existing")

def save_results_to_csv(results, output_csv):
    """
    Save results to CSV file
    """
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['sub_dataset', 'video_name', 'caption']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

def main():
    parser = argparse.ArgumentParser(description='Generate frame captions for Articulation_results dataset using Gemini API')
    parser.add_argument('--api-keys', required=True, nargs='+', 
                        help='One or more Gemini API keys (space-separated)')
    parser.add_argument('--dataset-path', type=str, 
                        default='/research/cbim/vast/sf895/code/wsg/output/Articulation_results',
                        help='Path to Articulation_results dataset')
    parser.add_argument('--output-csv', type=str,
                        default=None,
                        help='Output CSV file path (default: dataset_path/articulation_captions.csv)')
    parser.add_argument('--num-frames', type=int, default=3,
                        help='Number of frames to use for caption generation (default: 3)')
    parser.add_argument('--max-videos', type=int, default=None,
                        help='Maximum number of videos to process (default: all)')
    parser.add_argument('--max-workers', type=int, default=None,
                        help='Maximum number of concurrent workers (default: auto)')
    parser.add_argument('--skip-existing', action='store_true', default=True,
                        help='Skip videos that already have captions in CSV')
    parser.add_argument('--no-skip-existing', dest='skip_existing', action='store_false',
                        help='Regenerate all captions')
    
    args = parser.parse_args()
    
    # Setup API pool
    api_pool = APIPool(args.api_keys)
    
    # Set default output CSV path if not provided
    if args.output_csv is None:
        args.output_csv = os.path.join(args.dataset_path, 'articulation_captions.csv')
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("Articulation Dataset Frame Caption Generator")
    print(f"{'='*60}")
    print(f"Dataset path: {args.dataset_path}")
    print(f"Output CSV: {args.output_csv}")
    print(f"Number of frames per caption: {args.num_frames}")
    print(f"Number of API keys: {api_pool.size()}")
    print(f"Skip existing: {args.skip_existing}")
    if args.max_videos:
        print(f"Max videos: {args.max_videos}")
    if args.max_workers:
        print(f"Max workers: {args.max_workers}")
    print(f"{'='*60}")
    
    # Process the dataset
    process_articulation_dataset(
        articulation_results_path=args.dataset_path,
        api_pool=api_pool,
        output_csv=args.output_csv,
        num_frames=args.num_frames,
        skip_existing=args.skip_existing,
        max_videos=args.max_videos,
        max_workers=args.max_workers
    )
    
    print(f"\n{'='*60}")
    print("Processing completed!")
    print(f"Results saved to: {args.output_csv}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()