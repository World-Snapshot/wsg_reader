import argparse
import glob
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
from tqdm import tqdm


CHANNELS = (122, 123, 124)


def load_wsg_data(wsg_path):
    try:
        data = np.load(wsg_path, allow_pickle=True)
        if "wsg_data" in data.files:
            return data["wsg_data"]
        if "data" in data.files:
            return data["data"]
        raise KeyError(f"Unsupported WSG payload keys: {list(data.files)}")
    except Exception as exc:
        print(f"Failed to load WSG data {wsg_path}: {exc}")
        return None


def _rgb_to_uint8(rgb):
    """WSG is expected to be denormalized. Only support 0-1 input as a compatibility fallback."""
    rgb = rgb.astype(np.float32, copy=False)
    valid = np.any(rgb > -0.5, axis=2)
    rgb = np.where(valid[:, :, None], rgb, 0.0)
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=255.0, neginf=0.0)
    if rgb.size and float(np.max(rgb)) <= 1.5:
        rgb = rgb * 255.0
    return np.clip(rgb, 0.0, 255.0).astype(np.uint8)


def extract_xpose_pose_rgb(wsg_array):
    if wsg_array.ndim != 3 or wsg_array.shape[2] <= max(CHANNELS):
        raise ValueError(f"Expected HxWx>=125 WSG array, got {wsg_array.shape}")
    return _rgb_to_uint8(wsg_array[:, :, CHANNELS])


def extract_base_rgb(wsg_array):
    if wsg_array.ndim != 3 or wsg_array.shape[2] < 3:
        raise ValueError(f"Expected HxWx>=3 WSG array, got {wsg_array.shape}")
    return _rgb_to_uint8(wsg_array[:, :, :3])


def save_rgb(path, image):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(image, mode="RGB").save(path)


def process_single_wsg_file(args):
    wsg_path, output_dir, modes = args
    modes = set(modes or ["vis", "vis_overlay", "save"])
    try:
        base_name = os.path.splitext(os.path.basename(wsg_path))[0]
        wsg_array = load_wsg_data(wsg_path)
        if wsg_array is None:
            return False

        pose_rgb = extract_xpose_pose_rgb(wsg_array)
        success_count = 0

        if "vis" in modes or "all" in modes:
            vis_path = os.path.join(output_dir, f"export_xpose_pose_from_wsg_{base_name}_vis.png")
            save_rgb(vis_path, pose_rgb)
            print(f"    Saved XPose pose visualization: {vis_path}")
            success_count += 1

        if "vis_overlay" in modes or "all" in modes:
            base_rgb = extract_base_rgb(wsg_array)
            overlay = np.clip(base_rgb.astype(np.float32) * 0.55 + pose_rgb.astype(np.float32) * 0.45, 0, 255).astype(np.uint8)
            overlay_path = os.path.join(output_dir, f"export_xpose_pose_from_wsg_{base_name}_overlay.png")
            save_rgb(overlay_path, overlay)
            print(f"    Saved XPose pose overlay: {overlay_path}")
            success_count += 1

        if "save" in modes or "npz" in modes or "all" in modes:
            npz_path = os.path.join(output_dir, f"export_xpose_pose_from_wsg_{base_name}.npz")
            os.makedirs(os.path.dirname(npz_path), exist_ok=True)
            np.savez_compressed(npz_path, xpose_pose_rgb=pose_rgb, channels=np.asarray(CHANNELS, dtype=np.int16))
            print(f"    Saved XPose pose NPZ: {npz_path}")
            success_count += 1

        return success_count > 0
    except Exception as exc:
        print(f"  Failed to process {wsg_path}: {exc}")
        return False


def process_wsg_input(input_path, output_path=None, modes=None, max_workers=4):
    modes = modes or ["vis", "vis_overlay", "save"]
    if os.path.isfile(input_path) and input_path.endswith(".wsg"):
        output_path = output_path or os.path.dirname(input_path)
        os.makedirs(output_path, exist_ok=True)
        return process_single_wsg_file((input_path, output_path, modes))

    if os.path.isdir(input_path):
        output_path = output_path or os.path.join(os.path.dirname(input_path.rstrip("/")), "export_xpose_pose_from_wsg")
        os.makedirs(output_path, exist_ok=True)
        wsg_files = sorted(glob.glob(os.path.join(input_path, "*.wsg")))
        if not wsg_files:
            print(f"No WSG files found in {input_path}")
            return False
        tasks = [(wsg_path, output_path, modes) for wsg_path in wsg_files]
        if max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(tqdm(executor.map(process_single_wsg_file, tasks), total=len(tasks), desc="Extract XPose pose"))
        else:
            results = [process_single_wsg_file(task) for task in tqdm(tasks, desc="Extract XPose pose")]
        return any(results)

    print(f"Input path is neither a WSG file nor a directory: {input_path}")
    return False


def main():
    parser = argparse.ArgumentParser(description="Extract XPose pose visualization channels 122-124 from WSG.")
    parser.add_argument("input_path", help="WSG file or directory")
    parser.add_argument("output_path", nargs="?", default=None, help="Output directory")
    parser.add_argument("--modes", default="vis,vis_overlay,save", help="Comma-separated modes: vis,vis_overlay,save,npz,all")
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()
    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    ok = process_wsg_input(args.input_path, args.output_path, modes=modes, max_workers=args.max_workers)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
