#!/usr/bin/env python3
"""
Generate captions for MatSynth_results dataset using Gemini API
Processes *_original.png images in each video's crop_frame directory
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

def generate_caption_for_image(model, image_path, material_name):
    """
    Generate caption for a material texture image
    """
    try:
        # Load image
        img = Image.open(image_path)
        
        # Prepare prompt for material texture
        prompt = """Please provide a detailed description of this material texture in 2-3 sentences. 
        Focus on the material's surface characteristics, patterns, color, texture details, 
        and any distinguishing visual properties. Describe what type of material this appears to be."""
        
        # Generate content
        response = model.generate_content([img, prompt])
        
        return response.text.strip()
    except Exception as e:
        print(f"Error processing image for {material_name}: {e}")
        return None

def process_material_folder(model, material_dir, max_retries=3):
    """
    Process a single material folder and generate caption
    """
    material_name = os.path.basename(material_dir)
    crop_frame_dir = os.path.join(material_dir, 'crop_frame')
    
    if not os.path.exists(crop_frame_dir):
        print(f"No crop_frame directory found for {material_name}")
        return "No crop_frame directory found"
    
    # Find the *_original.png file
    original_files = glob.glob(os.path.join(crop_frame_dir, '*_original.png'))
    
    if not original_files:
        print(f"No *_original.png file found in {material_name}")
        return "No original file found"
    
    # Use the first original file found
    image_path = original_files[0]
    
    # Try to generate caption with retries
    for attempt in range(max_retries):
        try:
            caption = generate_caption_for_image(model, image_path, material_name)
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

def get_all_material_folders(matsynth_results_path):
    """
    Get all material folders from MatSynth_results/wsg directory
    """
    wsg_path = os.path.join(matsynth_results_path, 'wsg')
    
    if not os.path.exists(wsg_path):
        print(f"WSG directory not found: {wsg_path}")
        return []
    
    material_folders = []
    
    # Find all folders that contain crop_frame subdirectory
    for item in os.listdir(wsg_path):
        item_path = os.path.join(wsg_path, item)
        if os.path.isdir(item_path):
            crop_frame_path = os.path.join(item_path, 'crop_frame')
            if os.path.exists(crop_frame_path) and os.path.isdir(crop_frame_path):
                # Check if there's an *_original.png file
                if glob.glob(os.path.join(crop_frame_path, '*_original.png')):
                    material_folders.append({
                        'path': item_path,
                        'material_name': item
                    })
    
    return material_folders

def process_single_material(api_pool, material_info, max_retries=3):
    """Process a single material using the API pool"""
    model = api_pool.get_model()
    material_path = material_info['path']
    material_name = material_info['material_name']
    
    # Generate caption
    caption = process_material_folder(model, material_path, max_retries)
    
    return {
        'material_name': material_name,
        'caption': caption
    }

def process_matsynth_dataset(matsynth_results_path, api_pool, output_csv, 
                            skip_existing=True, max_materials=None, max_workers=None):
    """
    Process all materials in MatSynth_results dataset and save captions to CSV
    """
    # Get all material folders
    material_folders = get_all_material_folders(matsynth_results_path)
    
    if not material_folders:
        print(f"No material folders found in {matsynth_results_path}/wsg")
        return
    
    print(f"\nFound {len(material_folders)} material folders with crop_frame directories")
    
    # Limit number of materials if specified
    if max_materials:
        material_folders = material_folders[:max_materials]
        print(f"\nProcessing only first {max_materials} materials")
    
    # Check if CSV already exists and load existing captions
    existing_captions = {}
    if skip_existing and os.path.exists(output_csv):
        with open(output_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_captions[row['material_name']] = row['caption']
        print(f"\nFound {len(existing_captions)} existing captions")
    
    # Prepare list of materials to process
    materials_to_process = []
    results = []
    skipped = 0
    
    for material_info in material_folders:
        material_name = material_info['material_name']
        
        # Skip if already processed
        if skip_existing and material_name in existing_captions:
            results.append({
                'material_name': material_name,
                'caption': existing_captions[material_name]
            })
            skipped += 1
        else:
            materials_to_process.append(material_info)
    
    print(f"\nMaterials to process: {len(materials_to_process)}")
    
    # Determine number of workers
    if max_workers is None:
        max_workers = min(api_pool.size() * 2, 10)  # 2 threads per API key, max 10
    
    # Process materials concurrently
    if len(materials_to_process) > 0:
        print(f"Using {max_workers} workers with {api_pool.size()} API keys")
        
        # Create a thread pool
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_material = {
                executor.submit(process_single_material, api_pool, material_info): material_info
                for material_info in materials_to_process
            }
            
            # Process completed tasks with progress bar
            with tqdm(total=len(materials_to_process), desc="Processing materials") as pbar:
                completed = 0
                
                for future in as_completed(future_to_material):
                    material_info = future_to_material[future]
                    try:
                        result = future.result()
                        results.append(result)
                        completed += 1
                        pbar.update(1)
                        
                        # Save intermediate results every 20 materials
                        if completed % 20 == 0:
                            save_results_to_csv(results, output_csv)
                            
                    except Exception as e:
                        print(f"\nError processing {material_info['material_name']}: {e}")
                        results.append({
                            'material_name': material_info['material_name'],
                            'caption': f"Error: {str(e)}"
                        })
                        pbar.update(1)
    
    # Save final results
    save_results_to_csv(results, output_csv)
    print(f"\nCompleted! Processed {len(results) - skipped} new materials, skipped {skipped} existing")

def save_results_to_csv(results, output_csv):
    """
    Save results to CSV file
    """
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['material_name', 'caption']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

def main():
    parser = argparse.ArgumentParser(description='Generate captions for MatSynth_results dataset using Gemini API')
    parser.add_argument('--api-keys', required=True, nargs='+', 
                        help='One or more Gemini API keys (space-separated)')
    parser.add_argument('--dataset-path', type=str, 
                        default='/research/cbim/vast/sf895/code/wsg/output/MatSynth_results',
                        help='Path to MatSynth_results dataset')
    parser.add_argument('--output-csv', type=str,
                        default=None,
                        help='Output CSV file path (default: dataset_path/matsynth_captions.csv)')
    parser.add_argument('--max-materials', type=int, default=None,
                        help='Maximum number of materials to process (default: all)')
    parser.add_argument('--max-workers', type=int, default=None,
                        help='Maximum number of concurrent workers (default: auto)')
    parser.add_argument('--skip-existing', action='store_true', default=True,
                        help='Skip materials that already have captions in CSV')
    parser.add_argument('--no-skip-existing', dest='skip_existing', action='store_false',
                        help='Regenerate all captions')
    
    args = parser.parse_args()
    
    # Setup API pool
    api_pool = APIPool(args.api_keys)
    
    # Set default output CSV path if not provided
    if args.output_csv is None:
        args.output_csv = os.path.join(args.dataset_path, 'matsynth_captions.csv')
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("MatSynth Dataset Caption Generator")
    print(f"{'='*60}")
    print(f"Dataset path: {args.dataset_path}")
    print(f"Output CSV: {args.output_csv}")
    print(f"Number of API keys: {api_pool.size()}")
    print(f"Skip existing: {args.skip_existing}")
    if args.max_materials:
        print(f"Max materials: {args.max_materials}")
    if args.max_workers:
        print(f"Max workers: {args.max_workers}")
    print(f"{'='*60}")
    
    # Process the dataset
    process_matsynth_dataset(
        matsynth_results_path=args.dataset_path,
        api_pool=api_pool,
        output_csv=args.output_csv,
        skip_existing=args.skip_existing,
        max_materials=args.max_materials,
        max_workers=args.max_workers
    )
    
    print(f"\n{'='*60}")
    print("Processing completed!")
    print(f"Results saved to: {args.output_csv}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()