"""Utilities for visualizing decoded categorical WSG channels."""

from __future__ import annotations

import colorsys
import json
from pathlib import Path

import numpy as np
from scipy import ndimage


MASK_VALUE = -1.0
CHANNEL_111 = 111
CHANNEL_111_ALPHA = 3
CHANNEL_111_MIN_ID = 0
CHANNEL_195 = 195
CHANNEL_195_MIN_ID = 1
CHANNEL_195_MAX_ID = 26

AVIS_CATEGORY_COLORS = {
    1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255), 4: (255, 255, 0),
    5: (255, 0, 255), 6: (0, 255, 255), 7: (255, 128, 0), 8: (128, 255, 0),
    9: (128, 0, 255), 10: (255, 0, 128), 11: (255, 192, 128), 12: (128, 128, 255),
    13: (255, 255, 128), 14: (128, 255, 128), 15: (128, 0, 0), 16: (170, 255, 195),
    17: (128, 128, 0), 18: (255, 215, 180), 19: (0, 0, 128), 20: (128, 128, 128),
    21: (255, 255, 255), 22: (20, 120, 20), 23: (120, 20, 20), 24: (20, 20, 120),
    25: (120, 80, 20), 26: (20, 120, 120),
}


def _load_local_categories(name: str) -> dict[int, str]:
    path = Path(__file__).resolve().parent / f"tools/{name}/category_id_mapping.json"
    with path.open("r", encoding="utf-8") as handle:
        category_to_id = json.load(handle)["category_to_id"]
    return {int(class_id): str(label).strip().lower() for label, class_id in category_to_id.items()}


SHAPENET_LOCAL_ID_TO_CATEGORY = _load_local_categories("ShapeNet")
OBJAVERSE_LOCAL_ID_TO_CATEGORY = _load_local_categories("Objaverse")

SEMANTIC_FAMILIES = (
    ("metadata", ("_", "attribute", "evalset", "studymodel", "gallery", "scenedb", "unknown", "misc")),
    ("people", ("person", "human", "character", "anatom", "mannequin", "anthrop")),
    ("animals", ("animal", "bear", "bird", "butterfly", "cat", "cow", "dog", "donkey", "elephant", "fish", "giraffe", "horse", "insect", "lion", "rabbit", "sheep", "creature", "myth")),
    ("plants_food", ("plant", "vegetable", "fruit", "apple", "banana", "carrot", "orange", "food", "bread", "cake", "cereal", "chocolate", "cookie", "donut", "egg", "fries", "milk", "pizza", "sandwich")),
    ("seating", ("chair", "stool", "bench", "couch", "sofa", "recliner", "ottoman", "beanbag", "loveseat", "futon", "chaise")),
    ("beds", ("bed", "mattress", "pillow", "cradle", "crib", "sleeper", "trundle")),
    ("tables", ("table", "desk", "counter", "lectern", "pedestal")),
    ("storage", ("cabinet", "shelf", "armoire", "bookcase", "closet", "credenza", "dresser", "rack", "wardrobe", "chest", "sideboard")),
    ("architecture", ("architect", "room", "door", "window", "courtyard", "fireplace", "booth", "tent", "canopy")),
    ("lighting", ("lamp", "light", "candle")),
    ("electronics", ("computer", "phone", "camera", "monitor", "printer", "electronic", "console", "controller", "xbox", "playstation", "ps", "wii", "ipad", "ipod", "radio", "speaker", "headphone", "webcam", "calculator", "copier")),
    ("appliances", ("conditioner", "blender", "dryer", "washer", "dishwasher", "microwave", "oven", "toaster", "refrigerator", "coffeemaker", "kettle", "fan")),
    ("vessels", ("bottle", "bowl", "cup", "glass", "mug", "plate", "pan", "pot", "jug", "teapot", "basket", "bucket", "canister")),
    ("tools_weapons", ("tool", "weapon", "gun", "sword", "knife", "hammer", "screwdriver", "scissors", "broom", "flashlight", "key", "lock")),
    ("vehicles", ("vehicle", "airplane", "bicycle", "bus", "car", "helicopter", "motorcycle", "ship", "tank", "truck")),
    ("music", ("instrument", "guitar", "harp", "piano", "trumpet", "violin", "drum")),
    ("clothing", ("clothing", "shirt", "shoe", "hat", "cap", "armor", "helmet", "bag", "backpack", "purse", "wallet", "jewelry", "ring", "watch", "glasses", "accessor")),
    ("office_media", ("book", "paper", "pen", "pencil", "marker", "folder", "notepad", "letter", "stapler", "tape", "ruler", "board", "picture", "poster", "painting", "media")),
    ("decor_toys", ("decor", "sculpt", "statue", "vase", "toy", "game", "chess", "dart", "ball", "rubik", "snowman", "palette", "planet")),
)


def semantic_family(label: str) -> int:
    normalized = str(label).lower().replace(" ", "")
    for family_index, (_name, keywords) in enumerate(SEMANTIC_FAMILIES):
        if any(keyword.replace(" ", "") in normalized for keyword in keywords):
            return family_index
    return len(SEMANTIC_FAMILIES)



# WSG v1 coarse deployment ontology. Channels 111 and 136 keep separate
# storage contracts and narratives, but share this semantic backbone so both
# object classifiers avoid hundreds of extremely sparse fine labels.
COARSE_OBJECT_CATEGORY_NAMES = {
    0: "other_or_unknown",
    1: "people",
    2: "animals_plants_and_food",
    3: "furniture_and_storage",
    4: "architecture_and_lighting",
    5: "electronics_and_appliances",
    6: "vessels_and_containers",
    7: "tools_and_weapons",
    8: "vehicles",
    9: "personal_and_cultural_objects",
}
COARSE_OBJECT_CATEGORY_COLORS = {
    0: (128, 128, 128), 1: (230, 57, 70), 2: (42, 157, 143),
    3: (244, 162, 97), 4: (141, 153, 174), 5: (69, 123, 157),
    6: (233, 196, 106), 7: (131, 56, 236), 8: (17, 138, 178),
    9: (188, 108, 137),
}
_SEMANTIC_FAMILY_TO_COARSE_OBJECT = {
    "metadata": 0, "people": 1, "animals": 2, "plants_food": 2,
    "seating": 3, "beds": 3, "tables": 3, "storage": 3,
    "architecture": 4, "lighting": 4,
    "electronics": 5, "appliances": 5, "vessels": 6,
    "tools_weapons": 7, "vehicles": 8,
    "music": 9, "clothing": 9, "office_media": 9, "decor_toys": 9,
}

COARSE_MATERIAL_CATEGORY_NAMES = {
    0: "unknown",
    1: "mineral_ceramic_and_glass",
    2: "wood_and_paper",
    3: "textile_leather_fur_and_organic",
    4: "metal",
    5: "plastic_and_rubber",
}
COARSE_MATERIAL_CATEGORY_COLORS = {
    0: (128, 128, 128), 1: (112, 172, 187), 2: (166, 124, 82),
    3: (135, 164, 84), 4: (174, 184, 194), 5: (77, 142, 216),
}
_FINE_MATERIAL_TO_COARSE = {
    "unknown": 0,
    "brick": 1, "ceramic": 1, "concrete": 1, "granite": 1, "stone": 1,
    "glass": 1,
    "cardboard": 2, "paper": 2, "wallpaper": 2, "wood": 2,
    "carpet": 3, "fabric": 3, "fur": 3, "leather": 3,
    "foliage": 3, "food": 3,
    "metal": 4, "plastic": 5, "rubber": 5,
}


def coarse_object_category_id(label: str) -> int:
    """Map one fine object label to the shared 10-class WSG ontology."""
    family_index = semantic_family(label)
    if family_index >= len(SEMANTIC_FAMILIES):
        return 0
    family_name = SEMANTIC_FAMILIES[family_index][0]
    return int(_SEMANTIC_FAMILY_TO_COARSE_OBJECT.get(family_name, 0))


def source_category_label(category_id: int, source_path: str) -> str:
    """Resolve a dataset-local category ID without merging channel contracts."""
    path_text = str(source_path)
    if "ShapeNetSem_results" in path_text:
        return SHAPENET_LOCAL_ID_TO_CATEGORY.get(int(category_id), "unknown")
    if "Articulation_results" in path_text or "Objaverse" in path_text:
        return OBJAVERSE_LOCAL_ID_TO_CATEGORY.get(int(category_id), "unknown")
    shape_label = SHAPENET_LOCAL_ID_TO_CATEGORY.get(int(category_id))
    object_label = OBJAVERSE_LOCAL_ID_TO_CATEGORY.get(int(category_id))
    if shape_label and not object_label:
        return shape_label
    if object_label and not shape_label:
        return object_label
    return shape_label or object_label or "unknown"


def coarse_source_object_category_id(category_id: int, source_path: str) -> int:
    return coarse_object_category_id(source_category_label(category_id, source_path))


def _load_material_id_to_name() -> dict[int, str]:
    path = Path(__file__).resolve().parent / "tools/ShapeNet/material_id_mapping.json"
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {int(key): str(value).strip().lower() for key, value in payload["id_to_material"].items()}


MATERIAL_FINE_ID_TO_NAME = _load_material_id_to_name()


def coarse_material_category_id(material_id: int) -> int:
    label = MATERIAL_FINE_ID_TO_NAME.get(int(material_id), "unknown")
    return int(_FINE_MATERIAL_TO_COARSE.get(label, 0))


def coarse_material_category_map(
    wsg_array: np.ndarray, *, invalid_value: float = MASK_VALUE,
) -> np.ndarray:
    """Return channel 131 using the stable six-class material ontology."""
    data = np.asarray(wsg_array)
    source = np.asarray(data[..., 131], dtype=np.float32)
    output = np.full(source.shape, invalid_value, dtype=np.float32)
    valid = np.isfinite(source) & (source != invalid_value)
    rounded = np.rint(source).astype(np.int32, copy=False)
    for value in np.unique(rounded[valid]).tolist() if valid.any() else []:
        output[valid & (rounded == value)] = float(coarse_material_category_id(value))
    return output


def coarse_thing_category_map(
    wsg_array: np.ndarray, source_path: str, *, invalid_value: float = MASK_VALUE,
) -> np.ndarray:
    """Return channel 136 as Thing/Physical Object Category in 10 classes."""
    data = np.asarray(wsg_array)
    source = np.asarray(data[..., 136], dtype=np.float32)
    output = np.full(source.shape, invalid_value, dtype=np.float32)
    valid = np.isfinite(source) & (source != invalid_value)
    rounded = np.rint(source).astype(np.int32, copy=False)
    for value in np.unique(rounded[valid]).tolist() if valid.any() else []:
        coarse_id = coarse_source_object_category_id(value, source_path)
        output[valid & (rounded == value)] = float(coarse_id)
    return output



_GLOBAL_LABELS = sorted(
    set(SHAPENET_LOCAL_ID_TO_CATEGORY.values()) | set(OBJAVERSE_LOCAL_ID_TO_CATEGORY.values()),
    key=lambda label: (semantic_family(label), label),
)
CHANNEL_111_CATEGORY_TO_ID = {label: index for index, label in enumerate(_GLOBAL_LABELS)}
CHANNEL_111_ID_TO_CATEGORY = {index: label for label, index in CHANNEL_111_CATEGORY_TO_ID.items()}
CHANNEL_111_MAX_ID = len(CHANNEL_111_ID_TO_CATEGORY) - 1


def build_distinct_palette() -> dict[int, tuple[int, int, int]]:
    """Build a stable palette: nearby semantics share hue, families span the gamut."""
    family_count = len(SEMANTIC_FAMILIES) + 1
    members: dict[int, list[int]] = {}
    for class_id, label in CHANNEL_111_ID_TO_CATEGORY.items():
        members.setdefault(semantic_family(label), []).append(class_id)
    palette = {}
    for family_index, class_ids in members.items():
        for rank, class_id in enumerate(class_ids):
            fraction = (rank + 0.5) / max(len(class_ids), 1)
            hue = (family_index + 0.08 + 0.84 * fraction) / family_count
            saturation = 0.68 + 0.30 * ((rank * 3) % 11) / 10.0
            value = 0.72 + 0.28 * ((rank * 7) % 13) / 12.0
            red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
            palette[class_id] = tuple(int(round(channel * 255.0)) for channel in (red, green, blue))
    return palette


CHANNEL_111_COLORS = build_distinct_palette()


def rgb_black_background_foreground_mask(
    wsg_array: np.ndarray,
    *,
    invalid_value: float = MASK_VALUE,
    min_coverage: float = 0.000001,
    max_coverage: float = 0.95,
) -> np.ndarray:
    """Recover an object silhouette from an RGB render on a black background.

    Articulation WSG files do not contain channel-3 alpha. Their RGB render has
    a black background, including small floating-point residuals. Background is
    estimated from the image border and propagated only through similarly dark
    pixels, preserving enclosed black regions inside an otherwise visible
    object. Small isolated rendering noise is removed afterwards.
    """
    data = np.asarray(wsg_array)
    if data.ndim != 3 or data.shape[-1] < 3:
        raise ValueError(f"Expected HxWx256 WSG data, got {data.shape}")
    rgb = np.asarray(data[..., :3], dtype=np.float32)
    valid_rgb = np.all(np.isfinite(rgb), axis=2) & np.all(rgb != invalid_value, axis=2)
    if not np.any(valid_rgb):
        raise ValueError("RGB foreground extraction requires valid RGB pixels")

    # Articulation renders define the background as black. Estimating it
    # from the crop border fails when a large object touches every border or
    # fills the entire crop. Only the numeric RGB scale needs to be inferred.
    background_rgb = np.zeros(3, dtype=np.float32)
    rgb_scale = 1.0 if float(np.nanmax(np.abs(rgb[valid_rgb]))) <= 2.0 else 255.0
    tolerance = 2.0 / 255.0 * rgb_scale

    distance = np.linalg.norm(rgb - background_rgb[None, None, :], axis=2)
    background_candidate = (~valid_rgb) | (distance <= tolerance)
    border_seed = np.zeros(background_candidate.shape, dtype=bool)
    border_seed[0] = background_candidate[0]
    border_seed[-1] = background_candidate[-1]
    border_seed[:, 0] |= background_candidate[:, 0]
    border_seed[:, -1] |= background_candidate[:, -1]
    connected_background = ndimage.binary_propagation(
        border_seed,
        structure=np.ones((3, 3), dtype=bool),
        mask=background_candidate,
    )
    foreground = valid_rgb & ~connected_background
    foreground |= ndimage.binary_closing(
        foreground, structure=np.ones((3, 3), dtype=bool)
    )
    foreground = ndimage.binary_fill_holes(foreground)

    labels, component_count = ndimage.label(foreground, structure=np.ones((3, 3), dtype=bool))
    if component_count:
        component_sizes = np.bincount(labels.ravel())
        minimum_pixels = max(16, int(round(foreground.size * 0.0001)))
        keep = component_sizes >= minimum_pixels
        keep[0] = False
        foreground = keep[labels]

    labels, component_count = ndimage.label(
        foreground, structure=np.ones((3, 3), dtype=bool)
    )
    component_sizes = np.bincount(labels.ravel())[1:]
    foreground_pixels = int(np.count_nonzero(foreground))
    largest_fraction = (
        float(component_sizes.max()) / foreground_pixels
        if foreground_pixels and component_sizes.size
        else 0.0
    )
    if component_count > 128 and largest_fraction < 0.25:
        raise ValueError(
            "RGB-derived foreground is fragmented noise: "
            f"components={component_count}, largest_fraction={largest_fraction:.4f}"
        )

    coverage = float(np.mean(foreground))
    if not min_coverage <= coverage <= max_coverage:
        raise ValueError(
            f"RGB-derived foreground coverage {coverage:.6f} is outside "
            f"[{min_coverage:.6f}, {max_coverage:.6f}]"
        )
    return np.asarray(foreground, dtype=bool)


def channel195_semantic_map(
    wsg_array: np.ndarray,
    *,
    invalid_value: float = MASK_VALUE,
) -> np.ndarray:
    """Return channel 195 as discrete IDs with an explicit -1 background."""
    data = np.asarray(wsg_array)
    if data.ndim != 3 or data.shape[-1] <= CHANNEL_195:
        raise ValueError(f"Expected HxWx256 WSG data, got {data.shape}")
    source = np.asarray(data[..., CHANNEL_195], dtype=np.float32)
    output = np.full(source.shape, invalid_value, dtype=np.float32)
    valid = np.isfinite(source) & (source >= CHANNEL_195_MIN_ID) & (source <= CHANNEL_195_MAX_ID)
    output[valid] = np.rint(source[valid])
    return output


def dominant_category_id(
    values: np.ndarray,
    *,
    invalid_value: float = MASK_VALUE,
    min_class_id: int = CHANNEL_111_MIN_ID,
    max_class_id: int = CHANNEL_111_MAX_ID,
) -> int:
    """Return the most frequent valid integer category ID in a sparse plane."""
    data = np.asarray(values, dtype=np.float32)
    valid = np.isfinite(data) & (data != invalid_value)
    if not valid.any():
        raise ValueError("Channel 111 contains no valid category ID")
    rounded = np.rint(data[valid]).astype(np.int32, copy=False)
    in_range = (rounded >= int(min_class_id)) & (rounded <= int(max_class_id))
    if not in_range.any():
        observed = np.unique(rounded)[:10].tolist()
        raise ValueError(f"Channel 111 category IDs are outside [{min_class_id}, {max_class_id}]: {observed}")
    labels, counts = np.unique(rounded[in_range], return_counts=True)
    return int(labels[int(np.argmax(counts))])


def dense_channel111_category_map(
    wsg_array: np.ndarray,
    *,
    invalid_value: float = MASK_VALUE,
    require_alpha: bool = True,
    category_id: int | None = None,
    allow_category_mask_fallback: bool = False,
    allow_region_mask_fallback: bool = False,
    allow_rgb_mask_fallback: bool = False,
) -> np.ndarray:
    """Derive a dense channel-111 map without modifying the source WSG.

    Raw object WSG files store the category ID on sparse mesh records. This
    function extracts their dominant category and spreads it over the positive
    alpha foreground. Generated target-only WSG files have no alpha lane; when
    ``require_alpha`` is false their already-dense channel 111 is retained.
    """
    data = np.asarray(wsg_array)
    if data.ndim != 3 or data.shape[-1] <= CHANNEL_111:
        raise ValueError(f"Expected HxWx256 WSG data, got {data.shape}")
    category_plane = np.asarray(data[..., CHANNEL_111], dtype=np.float32)
    if category_id is None:
        category_id = dominant_category_id(category_plane, invalid_value=invalid_value)
    if not CHANNEL_111_MIN_ID <= int(category_id) <= CHANNEL_111_MAX_ID:
        raise ValueError(
            f"Channel 111 category ID is outside [0, {CHANNEL_111_MAX_ID}]: {category_id}"
        )
    alpha = np.asarray(data[..., CHANNEL_111_ALPHA], dtype=np.float32)
    alpha_mask = np.isfinite(alpha) & (alpha != invalid_value) & (alpha > 0.0)
    if alpha_mask.any():
        dense = np.full(category_plane.shape, invalid_value, dtype=np.float32)
        dense[alpha_mask] = float(category_id)
        return dense
    if allow_region_mask_fallback:
        # ShapeNet object records expose reliable dense spatial support in
        # object/material lanes even when RGBA alpha is absent or corrupted.
        for region_channel in (136, 131, 143):
            if data.shape[-1] <= region_channel:
                continue
            region = np.asarray(data[..., region_channel], dtype=np.float32)
            region_mask = np.isfinite(region) & (region != invalid_value)
            coverage = float(np.mean(region_mask))
            if 0.000001 < coverage < 0.95:
                dense = np.full(category_plane.shape, invalid_value, dtype=np.float32)
                dense[region_mask] = float(category_id)
                return dense
    if allow_rgb_mask_fallback:
        rgb_mask = rgb_black_background_foreground_mask(data, invalid_value=invalid_value)
        dense = np.full(category_plane.shape, invalid_value, dtype=np.float32)
        dense[rgb_mask] = float(category_id)
        return dense
    if allow_category_mask_fallback:
        category_mask = np.isfinite(category_plane) & (category_plane != invalid_value)
        if category_mask.any():
            dense = np.full(category_plane.shape, invalid_value, dtype=np.float32)
            dense[category_mask] = float(category_id)
            return dense
    if require_alpha:
        raise ValueError("Channel 111 dense encoding requires a positive channel-3 alpha mask")
    valid = np.isfinite(category_plane) & (category_plane != invalid_value)
    dense = np.full(category_plane.shape, invalid_value, dtype=np.float32)
    dense[valid] = np.clip(
        np.rint(category_plane[valid]), CHANNEL_111_MIN_ID, CHANNEL_111_MAX_ID
    )
    return dense



def coarse_asset_category_map(
    wsg_array: np.ndarray,
    source_path: str,
    *,
    invalid_value: float = MASK_VALUE,
    category_id: int | None = None,
) -> np.ndarray:
    """Return channel 111 as Object/Asset Category in the shared 10 classes.

    This channel describes the source asset identity. It intentionally remains
    distinct from channel 136, which classifies visible physical things.
    """
    data = np.asarray(wsg_array)
    if category_id is None:
        raw_id = dominant_category_id(
            data[..., CHANNEL_111],
            invalid_value=invalid_value,
            max_class_id=max(
                max(SHAPENET_LOCAL_ID_TO_CATEGORY, default=0),
                max(OBJAVERSE_LOCAL_ID_TO_CATEGORY, default=0),
            ),
        )
        category_id = coarse_source_object_category_id(raw_id, source_path)
    coarse_id = int(category_id)
    return dense_channel111_category_map(
        data,
        invalid_value=invalid_value,
        require_alpha=True,
        category_id=coarse_id,
        allow_region_mask_fallback=True,
        allow_category_mask_fallback=True,
        allow_rgb_mask_fallback=True,
    )



def categorical_ids_to_rgb(
    values: np.ndarray,
    colors: dict[int, tuple[int, int, int]] = CHANNEL_111_COLORS,
    *,
    invalid_value: float = MASK_VALUE,
) -> np.ndarray:
    """Map integer IDs to exact palette colors and invalid background to black."""
    data = np.asarray(values, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"Expected a 2D categorical map, got {data.shape}")
    valid = np.isfinite(data) & (data != invalid_value)
    labels = np.rint(data).astype(np.int32, copy=False)
    output = np.zeros((*data.shape, 3), dtype=np.uint8)
    for class_id, color in colors.items():
        output[valid & (labels == int(class_id))] = color
    return output


def categorical_rgb_to_ids(
    rgb: np.ndarray,
    colors: dict[int, tuple[int, int, int]] = CHANNEL_111_COLORS,
    *,
    invalid_value: float = MASK_VALUE,
    chunk_pixels: int = 32768,
) -> np.ndarray:
    """Decode RGB to the nearest discrete class color, with black as invalid."""
    image = np.asarray(rgb, dtype=np.float32)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HxWx3 RGB data, got {image.shape}")
    if image.size and float(np.nanmax(image)) <= 1.5:
        image = image * 255.0
    class_ids = np.asarray(sorted(colors), dtype=np.int32)
    palette = np.asarray([colors[int(class_id)] for class_id in class_ids], dtype=np.float32)
    anchors = np.concatenate([np.zeros((1, 3), dtype=np.float32), palette], axis=0)
    anchor_norms = np.sum(anchors * anchors, axis=1)
    flat = np.nan_to_num(image.reshape(-1, 3), nan=0.0, posinf=255.0, neginf=0.0)
    decoded = np.empty(flat.shape[0], dtype=np.float32)
    for start in range(0, flat.shape[0], int(chunk_pixels)):
        pixels = flat[start:start + int(chunk_pixels)]
        distances = (
            np.sum(pixels * pixels, axis=1, keepdims=True)
            + anchor_norms[None, :]
            - 2.0 * pixels @ anchors.T
        )
        nearest = np.argmin(distances, axis=1)
        labels = np.full(nearest.shape, invalid_value, dtype=np.float32)
        foreground = nearest > 0
        labels[foreground] = class_ids[nearest[foreground] - 1].astype(np.float32)
        decoded[start:start + len(pixels)] = labels
    return decoded.reshape(image.shape[:2])


def categorical_colormap_soft(
    values: np.ndarray,
    colors: dict[int, tuple[int, int, int]],
    *,
    invalid_value: float = MASK_VALUE,
) -> np.ndarray:
    """Map dense categorical values to colors with interpolation.

    Some WSG categorical channels are stored as integer class ids, but VAE
    decode returns continuous values. Hard equality checks only show pixels that
    happen to decode exactly to an integer. This function treats class ids as
    color anchors and linearly interpolates between neighboring class colors.

    Colors are returned in the same channel order as provided by ``colors``.
    """
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D categorical map, got {values.shape}")
    if not colors:
        raise ValueError("colors must not be empty")

    class_ids = np.array(sorted(colors), dtype=np.float32)
    palette = np.array([colors[int(class_id)] for class_id in class_ids], dtype=np.float32)

    data = values.astype(np.float32, copy=False)
    valid = np.isfinite(data) & (data != invalid_value)
    clipped = np.clip(data, float(class_ids[0]), float(class_ids[-1]))

    upper_indices = np.searchsorted(class_ids, clipped, side="left")
    upper_indices = np.clip(upper_indices, 0, len(class_ids) - 1)
    lower_indices = np.maximum(upper_indices - 1, 0)

    exact = class_ids[upper_indices] == clipped
    lower_indices = np.where(exact, upper_indices, lower_indices)

    lower_values = class_ids[lower_indices]
    upper_values = class_ids[upper_indices]
    denom = np.maximum(upper_values - lower_values, 1e-6)
    weight = ((clipped - lower_values) / denom)[..., None]

    mapped = palette[lower_indices] * (1.0 - weight) + palette[upper_indices] * weight
    output = np.zeros((*data.shape, 3), dtype=np.uint8)
    output[valid] = np.clip(mapped[valid], 0, 255).astype(np.uint8)
    return output


def rounded_class_distribution(
    values: np.ndarray,
    class_ids: list[int] | tuple[int, ...],
    *,
    invalid_value: float = MASK_VALUE,
) -> dict[int, int]:
    """Count decoded categorical values after nearest-class quantization."""
    data = values.astype(np.float32, copy=False)
    valid = np.isfinite(data) & (data != invalid_value)
    if not valid.any():
        return {}
    classes = np.array(sorted(class_ids), dtype=np.float32)
    rounded = np.rint(data[valid])
    rounded = np.clip(rounded, classes[0], classes[-1]).astype(np.int32)
    unique, counts = np.unique(rounded, return_counts=True)
    return {int(class_id): int(count) for class_id, count in zip(unique, counts)}
