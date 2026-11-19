#!/usr/bin/env python3
"""
Clean up failed caption entries and their corresponding folders
"""
import os
import csv
import shutil
import argparse
from pathlib import Path

def clean_failed_captions(csv_path, dataset_path, dry_run=False):
    """
    Remove failed caption entries from CSV and delete corresponding folders
    """
    # Read the CSV file
    rows_to_keep = []
    folders_to_delete = []
    failed_count = 0
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        
        for row in reader:
            if row['caption'] == 'Failed to generate caption':
                # Mark folder for deletion
                folder_path = os.path.join(
                    dataset_path, 
                    row['sub_dataset'], 
                    row['video_name']
                )
                folders_to_delete.append(folder_path)
                failed_count += 1
                print(f"Found failed: {row['sub_dataset']}/{row['video_name']}")
            else:
                # Keep this row
                rows_to_keep.append(row)
    
    print(f"\nTotal entries: {len(rows_to_keep) + failed_count}")
    print(f"Failed entries to remove: {failed_count}")
    print(f"Entries to keep: {len(rows_to_keep)}")
    
    if dry_run:
        print("\n[DRY RUN] Would delete the following folders:")
        for folder in folders_to_delete[:10]:  # Show first 10
            print(f"  {folder}")
        if len(folders_to_delete) > 10:
            print(f"  ... and {len(folders_to_delete) - 10} more folders")
        return
    
    # Ask for confirmation
    response = input(f"\nDelete {failed_count} failed entries and folders? (yes/no): ")
    if response.lower() != 'yes':
        print("Cancelled.")
        return
    
    # Delete folders
    deleted_folders = 0
    for folder in folders_to_delete:
        if os.path.exists(folder):
            try:
                shutil.rmtree(folder)
                deleted_folders += 1
                if deleted_folders % 100 == 0:
                    print(f"Deleted {deleted_folders} folders...")
            except Exception as e:
                print(f"Error deleting {folder}: {e}")
        else:
            print(f"Folder not found: {folder}")
    
    # Write cleaned CSV
    backup_path = csv_path + '.backup'
    print(f"\nBacking up original CSV to: {backup_path}")
    shutil.copy2(csv_path, backup_path)
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_to_keep)
    
    print(f"\nCleaning complete!")
    print(f"Deleted {deleted_folders} folders")
    print(f"CSV now has {len(rows_to_keep)} entries")

def main():
    parser = argparse.ArgumentParser(description='Clean failed caption entries and folders')
    parser.add_argument('--csv-path', type=str,
                        default='/research/cbim/vast/sf895/code/wsg/output/Articulation_results/articulation_captions.csv',
                        help='Path to the captions CSV file')
    parser.add_argument('--dataset-path', type=str,
                        default='/research/cbim/vast/sf895/code/wsg/output/Articulation_results',
                        help='Path to the dataset root directory')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be deleted without actually deleting')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_path):
        print(f"CSV file not found: {args.csv_path}")
        return
    
    if not os.path.exists(args.dataset_path):
        print(f"Dataset path not found: {args.dataset_path}")
        return
    
    clean_failed_captions(args.csv_path, args.dataset_path, args.dry_run)

if __name__ == '__main__':
    main()