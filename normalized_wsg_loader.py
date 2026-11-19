"""Safely prefer a normalized WSG only when its normalization is reversible."""

import json
import os
from pathlib import Path

import numpy as np


MASK_VALUE = -1.0


def write_normalization_sidecar(normalized_path, modality_config_path):
    """Mark a normalized WSG as reversible using a portable modality config."""
    normalized_path = Path(normalized_path).resolve()
    modality_config_path = Path(modality_config_path).resolve()
    sidecar_path = Path(str(normalized_path) + ".normalization.json")
    payload = {"modality_config_path": os.path.relpath(modality_config_path, sidecar_path.parent)}
    with sidecar_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return sidecar_path


def load_wsg_data(wsg_path):
    try:
        with np.load(wsg_path, allow_pickle=False) as data:
            if "wsg_data" in data:
                return np.asarray(data["wsg_data"])
            if "data" in data:
                return np.asarray(data["data"])
            if data.files:
                return np.asarray(data[data.files[0]])
    except Exception as exc:
        print(f"Failed to load WSG {wsg_path}: {exc}")
    return None


def _normalized_path(wsg_path):
    base_path = os.path.splitext(str(wsg_path))[0]
    path = Path(base_path + "_normalized.wsg")
    return path if path.exists() else None


def _load_sidecar(normalized_path):
    candidates = [
        Path(str(normalized_path) + ".normalization.json"),
        normalized_path.with_suffix(".normalization.json"),
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, dict):
                return payload, path
        except Exception as exc:
            print(f"  WARNING: failed to read normalization metadata {path}: {exc}")
    return None, None


def _channel_mode(metadata, channel):
    channel_modes = metadata.get("channel_modes")
    if isinstance(channel_modes, dict):
        mode = channel_modes.get(str(channel), channel_modes.get(channel))
    else:
        mode = metadata.get("normalization_mode")
    if mode is None and metadata.get("modality_config_path"):
        mode = _mode_from_modality_config(metadata, channel)
    mode = str(mode).lower() if mode is not None else None
    return mode if mode in {"linear", "log"} else None


def _parse_modality_channels(modality):
    channels = []
    for part in str(modality).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            channels.extend(range(int(start), int(end) + 1))
        else:
            channels.append(int(part))
    return channels


def _mode_from_modality_config(metadata, channel):
    config_path = _resolve_path(metadata["modality_config_path"], metadata["_sidecar_path"])
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        print(f"  WARNING: failed to read modality config {config_path}: {exc}")
        return None
    normalization = payload.get("normalization")
    if not isinstance(normalization, dict):
        return None
    default_mode = normalization.get("default_mode")
    for modality, mode in (normalization.get("modalities") or {}).items():
        if int(channel) in _parse_modality_channels(modality):
            return mode
    return default_mode


def _direct_channel_range(metadata, channel):
    ranges = metadata.get("channel_ranges")
    if not isinstance(ranges, dict):
        return None
    payload = ranges.get(str(channel), ranges.get(channel))
    if isinstance(payload, dict) and "min_val" in payload and "max_val" in payload:
        return float(payload["min_val"]), float(payload["max_val"])
    if isinstance(payload, (list, tuple)) and len(payload) == 2:
        return float(payload[0]), float(payload[1])
    return None


def _resolve_path(path, sidecar_path):
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = sidecar_path.parent / path
    return path.resolve()


def _cached_channel_range(metadata, sidecar_path, channel):
    cache_path = metadata.get("normalization_cache_path")
    dataset_paths = metadata.get("dataset_paths") or metadata.get("normalization_dataset_paths")
    if not cache_path or not isinstance(dataset_paths, list) or not dataset_paths:
        return None
    cache_path = _resolve_path(cache_path, sidecar_path)
    try:
        with cache_path.open("r", encoding="utf-8") as handle:
            cache = json.load(handle)
    except Exception as exc:
        print(f"  WARNING: failed to read normalization cache {cache_path}: {exc}")
        return None

    minimum = None
    maximum = None
    for dataset_path in dataset_paths:
        resolved = str(_resolve_path(dataset_path, sidecar_path))
        entry = cache.get(resolved, cache.get(str(dataset_path)))
        channel_info = (entry or {}).get("channels", {}).get(str(channel), {})
        normalization = channel_info.get("normalization")
        if not isinstance(normalization, dict):
            continue
        current_min = float(normalization["min_val"])
        current_max = float(normalization["max_val"])
        minimum = current_min if minimum is None else min(minimum, current_min)
        maximum = current_max if maximum is None else max(maximum, current_max)
    return None if minimum is None or maximum is None else (minimum, maximum)


def _channel_range(metadata, sidecar_path, channel):
    direct = _direct_channel_range(metadata, channel)
    if direct is not None:
        return direct
    if metadata.get("modality_config_path"):
        config_path = _resolve_path(metadata["modality_config_path"], sidecar_path)
        try:
            with config_path.open("r", encoding="utf-8") as handle:
                config = json.load(handle)
            payload = (config.get("normalization") or {}).get("channel_ranges", {}).get(
                str(channel)
            )
            if isinstance(payload, (list, tuple)) and len(payload) == 2:
                return float(payload[0]), float(payload[1])
        except Exception as exc:
            print(f"  WARNING: failed to read portable ranges from {config_path}: {exc}")
    return _cached_channel_range(metadata, sidecar_path, channel)


def _denormalize_channel(values, channel, mode, value_range):
    valid = np.isfinite(values) & (values != MASK_VALUE)
    output = values.astype(np.float32, copy=True)
    if channel in (0, 1, 2):
        output[valid] = np.clip(output[valid], 0.0, 1.0) * 255.0
        return output
    minimum, maximum = value_range
    span = max(maximum - minimum, 1e-6)
    clipped = np.clip(output[valid], 0.0, 1.0)
    if mode == "log":
        output[valid] = np.expm1(clipped * max(np.log1p(span), 1e-6)) + minimum
    else:
        output[valid] = clipped * span + minimum
    return output


def load_preferred_wsg_data(wsg_path, channels):
    """Load reversible normalized data when possible, otherwise load the input WSG."""
    normalized_path = _normalized_path(wsg_path)
    if normalized_path is None:
        return load_wsg_data(wsg_path), False

    metadata, sidecar_path = _load_sidecar(normalized_path)
    if metadata is None:
        print(
            f"  WARNING: ignoring {normalized_path.name}; no normalization sidecar, "
            "so linear/log mode is unknown"
        )
        return load_wsg_data(wsg_path), False
    metadata["_sidecar_path"] = sidecar_path

    specifications = {}
    for channel in sorted(set(int(channel) for channel in channels)):
        if channel in (0, 1, 2):
            specifications[channel] = (None, None)
            continue
        mode = _channel_mode(metadata, channel)
        value_range = _channel_range(metadata, sidecar_path, channel)
        if mode is None or value_range is None:
            print(
                f"  WARNING: ignoring {normalized_path.name}; channel {channel} "
                "does not have reversible mode/range metadata"
            )
            return load_wsg_data(wsg_path), False
        specifications[channel] = (mode, value_range)

    normalized = load_wsg_data(normalized_path)
    if normalized is None:
        return load_wsg_data(wsg_path), False
    restored = normalized.astype(np.float32, copy=True)
    for channel, (mode, value_range) in specifications.items():
        if channel >= restored.shape[-1]:
            continue
        restored[..., channel] = _denormalize_channel(
            restored[..., channel], channel, mode, value_range
        )
    print(f"  Using reversible normalized WSG: {normalized_path.name}")
    return restored, True
