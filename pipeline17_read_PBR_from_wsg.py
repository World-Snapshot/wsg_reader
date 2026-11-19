#!/usr/bin/env python3
"""
Read material/PBR WSG files and export dense-map visualizations.

This reader matches the regenerated MatSynth WSG layout:
- 0-2: rendered RGB
- 3: rendered alpha
- 36-38: normal map
- 143: legacy per-pixel material region ID
- 144-146: legacy ambient color
- 147-149: legacy diffuse color
- 150: legacy opacity
- 151-153: legacy specular color
- 154: legacy shininess
- 155-157: legacy irradiance
- 161-163: basecolor map
- 164-166: diffuse map
- 167: height map
- 168: roughness map
- 169: metallic map
- 170-172: specular map
- 173: opacity map

Unlike the old center-pixel reader, this version exports full dense maps,
overlay visualizations, per-map NPZ files, and summary statistics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from pipeline16_read_psy_from_wsg import (
    VISUALIZATION_CHANNEL_RANGES,
    VISUALIZATION_SPECS,
    apply_continuous_palette,
    categorical_color,
    normalize_scalar_for_display,
    refine_reader_valid_mask,
    render_transparency_brightness,
)


TRIPLET_SPECS = [
    ((36, 37, 38), "normal"),
    ((144, 145, 146), "ambient_color"),
    ((147, 148, 149), "diffuse_color"),
    ((151, 152, 153), "specular_color"),
    ((155, 156, 157), "irradiance"),
    ((161, 162, 163), "basecolor"),
    ((164, 165, 166), "diffuse"),
    ((170, 171, 172), "specular"),
]

SCALAR_SPECS = [
    (3, "alpha"),
    (143, "material_region_id"),
    (150, "legacy_opacity"),
    (154, "shininess"),
    (167, "height"),
    (168, "roughness"),
    (169, "metallic"),
    (173, "opacity"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Read MatSynth PBR WSG files as dense maps.")
    parser.add_argument("input_path", type=Path, help="WSG file, material directory, or dataset WSG root.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional explicit output directory. By default each material writes to visualization_analysis/.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing exported files.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N WSG files.")
    return parser.parse_args()


def load_wsg_array(path: Path) -> np.ndarray:
    data = np.load(path)
    if isinstance(data, np.ndarray):
        return data
    if "wsg_data" in data.files:
        return data["wsg_data"]
    if "arr_0" in data.files:
        return data["arr_0"]
    if "data" in data.files:
        return data["data"]
    raise KeyError(f"Unsupported WSG payload keys in {path}: {list(data.files)}")


def discover_wsg_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    direct = sorted(p for p in input_path.glob("*.wsg") if p.is_file())
    if direct:
        return direct
    nested = sorted(p for p in input_path.glob("*/*.wsg") if p.is_file())
    return nested


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def valid_triplet_mask(arr: np.ndarray, channels: tuple[int, int, int]) -> np.ndarray:
    return np.all(arr[:, :, list(channels)] != -1, axis=2)


def valid_scalar_mask(arr: np.ndarray, channel: int) -> np.ndarray:
    return arr[:, :, channel] != -1


def make_base_rgb(arr: np.ndarray) -> np.ndarray:
    rgb = np.where(arr[:, :, 0:3] == -1, 1.0, arr[:, :, 0:3].astype(np.float32))
    return np.clip(rgb, 0.0, 1.0)


def save_rgb(path: Path, rgb: np.ndarray):
    Image.fromarray(np.clip(rgb * 255.0, 0, 255).astype(np.uint8), mode="RGB").save(path)


def save_rgba(path: Path, rgb: np.ndarray, alpha: np.ndarray):
    rgba = np.dstack([
        np.clip(rgb * 255.0, 0, 255).astype(np.uint8),
        np.clip(alpha * 255.0, 0, 255).astype(np.uint8),
    ])
    Image.fromarray(rgba, mode="RGBA").save(path)


def scalar_to_heatmap(normalized: np.ndarray) -> np.ndarray:
    normalized = np.clip(normalized, 0.0, 1.0)
    r = np.clip(1.5 * normalized - 0.1, 0.0, 1.0)
    g = np.clip(1.5 - 2.0 * np.abs(normalized - 0.5), 0.0, 1.0)
    b = np.clip(1.2 - 1.4 * normalized, 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def blend_overlay(base_rgb: np.ndarray, color_rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    overlay = base_rgb.copy()
    overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * color_rgb[mask]
    return np.clip(overlay, 0.0, 1.0)


def normalize_normal_map(normal_rgb: np.ndarray) -> np.ndarray:
    if normal_rgb.size == 0:
        return normal_rgb
    norm = normal_rgb.astype(np.float32).copy()
    lengths = np.linalg.norm(norm, axis=-1, keepdims=True)
    norm = norm / np.maximum(lengths, 1e-8)
    vis = norm * 0.5 + 0.5
    return np.clip(vis, 0.0, 1.0)


def scalar_stats(values: np.ndarray) -> dict:
    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "is_uniform": bool(np.allclose(values, values.flat[0], atol=1e-6)),
    }


def triplet_stats(values: np.ndarray) -> dict:
    return {
        "count": int(values.shape[0]),
        "min": np.min(values, axis=0).astype(np.float32).tolist(),
        "max": np.max(values, axis=0).astype(np.float32).tolist(),
        "mean": np.mean(values, axis=0).astype(np.float32).tolist(),
        "median": np.median(values, axis=0).astype(np.float32).tolist(),
        "is_uniform": bool(np.allclose(values, values[0], atol=1e-6)),
    }


def export_scalar_map(
    arr: np.ndarray,
    base_rgb: np.ndarray,
    output_dir: Path,
    stem: str,
    channel: int,
    slug: str,
    overwrite: bool,
) -> dict | None:
    value_map = arr[:, :, channel].astype(np.float32)
    mask = valid_scalar_mask(arr, channel)
    mask = refine_reader_valid_mask(value_map, mask, channel)
    if not np.any(mask):
        return None

    valid_values = value_map[mask]
    vmin = float(np.min(valid_values))
    vmax = float(np.max(valid_values))
    channel_spec = VISUALIZATION_SPECS.get(int(channel), {})
    if channel_spec.get("type") == "categorical":
        heatmap_u8 = np.zeros((*value_map.shape, 3), dtype=np.uint8)
        category_ids = np.rint(value_map).astype(np.int64)
        for category_id in np.unique(category_ids[mask]):
            heatmap_u8[mask & (category_ids == category_id)] = categorical_color(
                channel, int(category_id)
            )
        heatmap = heatmap_u8.astype(np.float32) / 255.0
        display_range = None
    elif int(channel) == 150:
        heatmap = render_transparency_brightness(value_map, mask)
        display_range = (0.0, 1.0)
    else:
        normalized, display_range = normalize_scalar_for_display(
            value_map,
            mask,
            VISUALIZATION_CHANNEL_RANGES.get(int(channel)),
        )
        heatmap = apply_continuous_palette(normalized, channel, mask).astype(np.float32) / 255.0
    overlay = blend_overlay(base_rgb, heatmap, mask)

    prefix = output_dir / f"{stem}_channel{channel}_{slug}"
    files = {
        "map_png": prefix.with_name(prefix.name + "_map.png"),
        "overlay_png": prefix.with_name(prefix.name + "_overlay.png"),
        "npz": prefix.with_name(prefix.name + "_map.npz"),
        "stats": prefix.with_name(prefix.name + "_stats.json"),
    }
    if overwrite or not files["map_png"].exists():
        save_rgb(files["map_png"], heatmap)
    if overwrite or not files["overlay_png"].exists():
        save_rgb(files["overlay_png"], overlay)
    if overwrite or not files["npz"].exists():
        np.savez_compressed(files["npz"], channel_map=value_map[:, :, None].astype(np.float32))
    stats = {
        "channel": channel,
        "slug": slug,
        "valid_pixels": int(np.sum(mask)),
        "coverage_ratio": float(np.mean(mask)),
        **scalar_stats(valid_values),
    }
    if overwrite or not files["stats"].exists():
        files["stats"].write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    return stats


def export_triplet_map(
    arr: np.ndarray,
    base_rgb: np.ndarray,
    output_dir: Path,
    stem: str,
    channels: tuple[int, int, int],
    slug: str,
    overwrite: bool,
) -> dict | None:
    mask = valid_triplet_mask(arr, channels)
    if not np.any(mask):
        return None

    triplet = arr[:, :, list(channels)].astype(np.float32)
    vis = np.zeros_like(base_rgb, dtype=np.float32)
    valid_triplet = np.clip(triplet[mask], 0.0, 1.0)
    if slug == "normal":
        vis[mask] = normalize_normal_map(valid_triplet)
    else:
        vis[mask] = valid_triplet
    overlay = blend_overlay(base_rgb, vis, mask)

    prefix = output_dir / f"{stem}_channel{channels[0]}_{channels[-1]}_{slug}"
    files = {
        "rgb_png": prefix.with_name(prefix.name + "_rgb.png"),
        "overlay_png": prefix.with_name(prefix.name + "_overlay.png"),
        "npz": prefix.with_name(prefix.name + "_map.npz"),
        "stats": prefix.with_name(prefix.name + "_stats.json"),
    }
    if overwrite or not files["rgb_png"].exists():
        save_rgb(files["rgb_png"], vis)
    if overwrite or not files["overlay_png"].exists():
        save_rgb(files["overlay_png"], overlay)
    if overwrite or not files["npz"].exists():
        np.savez_compressed(files["npz"], channel_map=triplet.astype(np.float32))
    stats = {
        "channels": list(channels),
        "slug": slug,
        "valid_pixels": int(np.sum(mask)),
        "coverage_ratio": float(np.mean(mask)),
        **triplet_stats(triplet[mask]),
    }
    if overwrite or not files["stats"].exists():
        files["stats"].write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    return stats


def export_front_render(arr: np.ndarray, output_dir: Path, stem: str, overwrite: bool) -> dict:
    base_rgb = make_base_rgb(arr)
    alpha_mask = valid_scalar_mask(arr, 3)
    alpha = np.zeros(arr.shape[:2], dtype=np.float32)
    if np.any(alpha_mask):
        alpha[alpha_mask] = np.clip(arr[:, :, 3][alpha_mask].astype(np.float32), 0.0, 1.0)

    rgb_path = output_dir / f"{stem}_front_rgb.png"
    alpha_path = output_dir / f"{stem}_front_alpha.png"
    rgba_path = output_dir / f"{stem}_front_rgba.png"
    if overwrite or not rgb_path.exists():
        save_rgb(rgb_path, base_rgb)
    if overwrite or not alpha_path.exists():
        save_rgb(alpha_path, np.repeat(alpha[:, :, None], 3, axis=2))
    if overwrite or not rgba_path.exists():
        save_rgba(rgba_path, base_rgb, alpha)

    return {
        "rgb_valid_pixels": int(np.sum(np.any(arr[:, :, 0:3] != -1, axis=2))),
        "alpha_valid_pixels": int(np.sum(alpha_mask)),
        "alpha_min": float(np.min(alpha)) if np.any(alpha_mask) else -1.0,
        "alpha_max": float(np.max(alpha)) if np.any(alpha_mask) else -1.0,
    }


def summarize_active_controls(arr: np.ndarray) -> dict:
    active = {}
    for channels, slug in TRIPLET_SPECS:
        mask = valid_triplet_mask(arr, channels)
        active[slug] = bool(np.any(mask))
    for channel, slug in SCALAR_SPECS:
        mask = valid_scalar_mask(arr, channel)
        active[slug] = bool(np.any(mask))
    active["basecolor_diffuse_overlap"] = bool(
        np.any(valid_triplet_mask(arr, (161, 162, 163)) & valid_triplet_mask(arr, (164, 165, 166)))
    )
    return active


def write_summary(output_dir: Path, stem: str, summary: dict, overwrite: bool):
    path = output_dir / f"{stem}_summary.json"
    if overwrite or not path.exists():
        path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def export_single_wsg(wsg_path: Path, output_dir: Path, overwrite: bool) -> dict:
    arr = load_wsg_array(wsg_path)
    stem = wsg_path.stem
    ensure_dir(output_dir)

    base_rgb = make_base_rgb(arr)
    front_stats = export_front_render(arr, output_dir, stem, overwrite)

    exported = {"triplets": {}, "scalars": {}}
    for channels, slug in TRIPLET_SPECS:
        stats = export_triplet_map(arr, base_rgb, output_dir, stem, channels, slug, overwrite)
        if stats is not None:
            exported["triplets"][slug] = stats
    for channel, slug in SCALAR_SPECS:
        stats = export_scalar_map(arr, base_rgb, output_dir, stem, channel, slug, overwrite)
        if stats is not None:
            exported["scalars"][slug] = stats

    summary = {
        "wsg_path": str(wsg_path),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "front": front_stats,
        "active_controls": summarize_active_controls(arr),
        "triplets": {k: {"valid_pixels": v["valid_pixels"], "is_uniform": v["is_uniform"]} for k, v in exported["triplets"].items()},
        "scalars": {k: {"valid_pixels": v["valid_pixels"], "is_uniform": v["is_uniform"], "min": v["min"], "max": v["max"]} for k, v in exported["scalars"].items()},
    }
    write_summary(output_dir, stem, summary, overwrite)
    return summary


def default_output_dir_for(wsg_path: Path) -> Path:
    return wsg_path.parent / "visualization_analysis"


def main():
    args = parse_args()
    wsg_files = discover_wsg_files(args.input_path)
    if args.limit is not None:
        wsg_files = wsg_files[: args.limit]
    if not wsg_files:
        raise SystemExit(f"No WSG files found under: {args.input_path}")

    summaries = []
    for idx, wsg_path in enumerate(wsg_files, start=1):
        output_dir = args.output_dir if args.output_dir is not None else default_output_dir_for(wsg_path)
        print(f"[{idx}/{len(wsg_files)}] Exporting {wsg_path.name} -> {output_dir}")
        try:
            summary = export_single_wsg(wsg_path, output_dir, args.overwrite)
            summaries.append(summary)
        except Exception as exc:
            print(f"   ❌ Failed: {exc}")

    if len(wsg_files) > 1:
        batch_summary_path = (args.output_dir or args.input_path) / "pipeline17_read_summary.json"
        ensure_dir(batch_summary_path.parent)
        batch_summary_path.write_text(
            json.dumps(
                {
                    "processed": len(summaries),
                    "requested": len(wsg_files),
                    "items": summaries,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"Saved batch summary: {batch_summary_path}")


if __name__ == "__main__":
    main()
