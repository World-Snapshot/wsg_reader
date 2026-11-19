#!/usr/bin/env python3
import os
import csv
import time
import glob
from pathlib import Path
import argparse
from tqdm import tqdm
import google.generativeai as genai

def setup_gemini_api(api_key):
    """
    Configure Gemini API with the provided API key
    """
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-1.5-flash')

def get_video_size_mb(video_path):
    """
    Get the size of video file in MB
    """
    return os.path.getsize(video_path) / (1024 * 1024)

def generate_caption_inline(model, video_path):
    """
    Generate caption for small video files (<20MB) using inline method
    """
    try:
        with open(video_path, 'rb') as video_file:
            video_bytes = video_file.read()
        
        response = model.generate_content([
            {
                "mime_type": "video/mp4",
                "data": video_bytes
            },
            "Please provide a detailed description of this video in 2-3 sentences. Focus on the main subject, actions, and visual elements."
        ])
        
        return response.text.strip()
    except Exception as e:
        print(f"Error processing video inline: {e}")
        return None

def generate_caption_uploaded(model, video_path):
    """
    Generate caption for large video files using File API
    """
    try:
        # Upload the video file
        video_file = genai.upload_file(path=video_path)
        
        # Wait for file to be processed
        while video_file.state.name == "PROCESSING":
            print(".", end="", flush=True)
            time.sleep(2)
            video_file = genai.get_file(video_file.name)
        
        if video_file.state.name == "FAILED":
            raise ValueError(f"Video processing failed: {video_file.state.name}")
        
        # Generate content with the uploaded file
        response = model.generate_content([
            video_file,
            "Please provide a detailed description of this video in 2-3 sentences. Focus on the main subject, actions, and visual elements."
        ])
        
        # Delete the uploaded file to save space
        genai.delete_file(video_file.name)
        
        return response.text.strip()
    except Exception as e:
        print(f"Error processing uploaded video: {e}")
        return None

def process_video(model, video_path, max_retries=3):
    """
    Process a single video and generate caption
    """
    video_size_mb = get_video_size_mb(video_path)
    
    for attempt in range(max_retries):
        try:
            if video_size_mb < 20:  # Use inline method for small files
                caption = generate_caption_inline(model, video_path)
            else:  # Use upload method for larger files
                caption = generate_caption_uploaded(model, video_path)
            
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

def process_dataset(dataset_path, model, output_csv, skip_existing=True):
    """
    Process all videos in MMFi dataset and save captions to CSV
    """
    # Find all source_video_final.mp4 files in MMFi dataset structure
    # MMFi structure: MMFi_Dataset_results/E01S01A01/source_video_final.mp4
    video_pattern = os.path.join(dataset_path, '*', 'source_video_final.mp4')
    video_files = sorted(glob.glob(video_pattern))
    
    if not video_files:
        # Also try looking for other video patterns in case the structure varies
        alt_patterns = [
            os.path.join(dataset_path, '*', '*.mp4'),
            os.path.join(dataset_path, '*', 'video', '*.mp4'),
            os.path.join(dataset_path, '*', 'rgb', '*.mp4')
        ]
        for pattern in alt_patterns:
            video_files = sorted(glob.glob(pattern))
            if video_files:
                print(f"Found videos using pattern: {pattern}")
                break
    
    if not video_files:
        print(f"No video files found in {dataset_path}")
        print(f"Tried patterns: {video_pattern} and alternatives")
        return
    
    print(f"\nFound {len(video_files)} videos in {os.path.basename(dataset_path)}")
    
    # Check if CSV already exists and load existing captions
    existing_captions = {}
    if skip_existing and os.path.exists(output_csv):
        with open(output_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_captions[row['video_name']] = row['caption']
        print(f"Found {len(existing_captions)} existing captions")
    
    # Process videos
    results = []
    skipped = 0
    
    for video_path in tqdm(video_files, desc=f"Processing {os.path.basename(dataset_path)}"):
        # Extract experiment name (e.g., E01S01A01) - parent directory name
        video_name = os.path.basename(os.path.dirname(video_path))
        
        # Skip if already processed
        if skip_existing and video_name in existing_captions:
            results.append({
                'video_name': video_name,
                'caption': existing_captions[video_name]
            })
            skipped += 1
            continue
        
        # Generate caption
        print(f"\nProcessing: {video_name}")
        caption = process_video(model, video_path)
        
        results.append({
            'video_name': video_name,
            'caption': caption
        })
        
        # Save intermediate results every 10 videos
        if len(results) % 10 == 0:
            save_results_to_csv(results, output_csv)
        
        # Rate limiting - wait between requests
        time.sleep(2)
    
    # Save final results
    save_results_to_csv(results, output_csv)
    print(f"\nCompleted! Processed {len(results) - skipped} new videos, skipped {skipped} existing")

def save_results_to_csv(results, output_csv):
    """
    Save results to CSV file
    """
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['video_name', 'caption']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

def main():
    parser = argparse.ArgumentParser(description='Generate video captions for MMFi dataset using Gemini API')
    parser.add_argument('--api-key', required=True, help='Gemini API key')
    parser.add_argument('--dataset-path', type=str, 
                        default='/research/cbim/vast/sf895/code/wsg/output/MMFi_Dataset_results',
                        help='Path to MMFi dataset')
    parser.add_argument('--output-csv', type=str,
                        default=None,
                        help='Output CSV file path (default: dataset_path/video_captions.csv)')
    parser.add_argument('--skip-existing', action='store_true', default=True,
                        help='Skip videos that already have captions in CSV')
    parser.add_argument('--no-skip-existing', dest='skip_existing', action='store_false',
                        help='Regenerate all captions')
    
    args = parser.parse_args()
    
    # Setup Gemini API
    model = setup_gemini_api(args.api_key)
    
    # Set default output CSV path if not provided
    if args.output_csv is None:
        args.output_csv = os.path.join(args.dataset_path, 'video_captions.csv')
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("MMFi Video Caption Generator")
    print(f"{'='*60}")
    print(f"Dataset path: {args.dataset_path}")
    print(f"Output CSV: {args.output_csv}")
    print(f"Skip existing: {args.skip_existing}")
    print(f"{'='*60}")
    
    # Process the dataset
    process_dataset(
        dataset_path=args.dataset_path,
        model=model,
        output_csv=args.output_csv,
        skip_existing=args.skip_existing
    )
    
    print(f"\n{'='*60}")
    print("Processing completed!")
    print(f"Results saved to: {args.output_csv}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()