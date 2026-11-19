# RealCam-Vid Dataset Download Tool

This tool downloads the RealCam-Vid dataset, which contains:
- ~100K video clips with dynamic scenes and camera movements
- Metric-scale camera annotations processed by MonST3R
- Total size: ~227GB (DL3DV-10K: 74G, RealEstate10K: 75G, MiraData9K: 78G)

## Usage

### Method 1: Using Python script (Recommended)

```bash
# Run the download script
python download_realcam_vid.py

# Or specify a custom download path
python download_realcam_vid.py --data-root /path/to/download

# To verify an existing download
python download_realcam_vid.py --verify-only
```

The script will:
1. Download all dataset files using huggingface-cli
2. Automatically unzip the downloaded files
3. Verify the download completeness

### Dataset Structure

After download and extraction, you'll have:
```
RealCam-Vid/
├── RealCam-Vid_train.npz    # Training metadata (~100K clips)
├── RealCam-Vid_test.npz     # Test metadata (5K clips)
├── DL3DV-10K/              # Indoor/outdoor scenes
├── RealEstate10K/          # Real estate videos
└── MiraData/               # Game renderings, sports, FPV, etc.
```

### Metadata Format

Each NPZ file contains:
- `video_path`: Relative path to video clip
- `short_caption`: Short caption (≤77 tokens)
- `long_caption`: Long caption (≤226 tokens)
- `camera_intrinsics`: [fx, fy, cx, cy] normalized by w/h
- `camera_extrinsics`: 4x4 w2c matrices
- `align_factor`: Scale factor for metric alignment
- `camera_scale`: Maximum L2 distance of camera positions
- `vtss_score`: Video training suitability score

### Note

The download uses HuggingFace mirror (hf-mirror.com) for faster speeds in some regions.
Make sure you have enough disk space (~300GB recommended for download + extraction).