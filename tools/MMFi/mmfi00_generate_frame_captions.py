#!/usr/bin/env python3
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

def setup_gemini_api(api_key):
    """
    Configure Gemini API with the provided API key
    """
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-1.5-flash')

def generate_caption_for_frames(model, frame_paths, experiment_name):
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
            prompt = "Please provide a detailed description of this image in 2-3 sentences. Focus on the person's actions, pose, surroundings, and any visible objects."
        else:
            prompt = f"These are {len(frame_paths)} sequential frames from a video. Please provide a detailed description in 2-3 sentences about what's happening across these frames. Focus on the person's movements, actions, and environment."
        
        # Generate content
        content = images + [prompt]
        response = model.generate_content(content)
        
        return response.text.strip()
    except Exception as e:
        print(f"Error processing frames for {experiment_name}: {e}")
        return None

def process_experiment(model, experiment_dir, num_frames=3, max_retries=3):
    """
    Process a single experiment folder and generate caption
    """
    experiment_name = os.path.basename(experiment_dir)
    crop_frame_dir = os.path.join(experiment_dir, 'crop_frame')
    
    if not os.path.exists(crop_frame_dir):
        print(f"No crop_frame directory found for {experiment_name}")
        return "No crop_frame directory found"
    
    # Get all frame files
    frame_files = sorted([f for f in os.listdir(crop_frame_dir) if f.endswith('.png')])
    
    if not frame_files:
        print(f"No frames found in {experiment_name}")
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
        selected_frames = frame_files
    
    frame_paths = [os.path.join(crop_frame_dir, f) for f in selected_frames]
    
    # Try to generate caption with retries
    for attempt in range(max_retries):
        try:
            caption = generate_caption_for_frames(model, frame_paths, experiment_name)
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

def process_dataset(dataset_path, model, output_csv, num_frames=3, skip_existing=True, max_experiments=None):
    """
    Process all experiments in MMFi dataset and save captions to CSV
    """
    # Find all experiment folders (E01S01A01 format)
    experiment_folders = sorted([
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d)) and d.startswith('E')
    ])
    
    if not experiment_folders:
        print(f"No experiment folders found in {dataset_path}")
        return
    
    print(f"\nFound {len(experiment_folders)} experiments in {os.path.basename(dataset_path)}")
    
    # Limit number of experiments if specified
    if max_experiments:
        experiment_folders = experiment_folders[:max_experiments]
        print(f"Processing only first {max_experiments} experiments")
    
    # Check if CSV already exists and load existing captions
    existing_captions = {}
    if skip_existing and os.path.exists(output_csv):
        with open(output_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_captions[row['video_name']] = row['caption']
        print(f"Found {len(existing_captions)} existing captions")
    
    # Process experiments
    results = []
    skipped = 0
    
    for experiment_name in tqdm(experiment_folders, desc="Processing experiments"):
        # Skip if already processed
        if skip_existing and experiment_name in existing_captions:
            results.append({
                'video_name': experiment_name,
                'caption': existing_captions[experiment_name]
            })
            skipped += 1
            continue
        
        # Generate caption
        experiment_dir = os.path.join(dataset_path, experiment_name)
        print(f"\nProcessing: {experiment_name}")
        caption = process_experiment(model, experiment_dir, num_frames)
        
        results.append({
            'video_name': experiment_name,
            'caption': caption
        })
        
        # Save intermediate results every 10 experiments
        if len(results) % 10 == 0:
            save_results_to_csv(results, output_csv)
        
        # Rate limiting - wait between requests
        time.sleep(2)
    
    # Save final results
    save_results_to_csv(results, output_csv)
    print(f"\nCompleted! Processed {len(results) - skipped} new experiments, skipped {skipped} existing")

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
    parser = argparse.ArgumentParser(description='Generate frame captions for MMFi dataset using Gemini API')
    parser.add_argument('--api-key', required=True, help='Gemini API key')
    parser.add_argument('--dataset-path', type=str, 
                        default='/research/cbim/vast/sf895/code/wsg/output/MMFi_Dataset_results',
                        help='Path to MMFi dataset')
    parser.add_argument('--output-csv', type=str,
                        default=None,
                        help='Output CSV file path (default: dataset_path/frame_captions.csv)')
    parser.add_argument('--num-frames', type=int, default=3,
                        help='Number of frames to use for caption generation (default: 3)')
    parser.add_argument('--max-experiments', type=int, default=None,
                        help='Maximum number of experiments to process (default: all)')
    parser.add_argument('--skip-existing', action='store_true', default=True,
                        help='Skip experiments that already have captions in CSV')
    parser.add_argument('--no-skip-existing', dest='skip_existing', action='store_false',
                        help='Regenerate all captions')
    
    args = parser.parse_args()
    
    # Setup Gemini API
    model = setup_gemini_api(args.api_key)
    
    # Set default output CSV path if not provided
    if args.output_csv is None:
        args.output_csv = os.path.join(args.dataset_path, 'frame_captions.csv')
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("MMFi Frame Caption Generator")
    print(f"{'='*60}")
    print(f"Dataset path: {args.dataset_path}")
    print(f"Output CSV: {args.output_csv}")
    print(f"Number of frames per caption: {args.num_frames}")
    print(f"Skip existing: {args.skip_existing}")
    if args.max_experiments:
        print(f"Max experiments: {args.max_experiments}")
    print(f"{'='*60}")
    
    # Process the dataset
    process_dataset(
        dataset_path=args.dataset_path,
        model=model,
        output_csv=args.output_csv,
        num_frames=args.num_frames,
        skip_existing=args.skip_existing,
        max_experiments=args.max_experiments
    )
    
    print(f"\n{'='*60}")
    print("Processing completed!")
    print(f"Results saved to: {args.output_csv}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()