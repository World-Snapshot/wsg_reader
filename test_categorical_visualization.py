import numpy as np

from categorical_visualization import (
    dense_channel111_category_map,
    rgb_black_background_foreground_mask,
)


def synthetic_wsg() -> np.ndarray:
    data = np.full((64, 64, 256), -1.0, dtype=np.float32)
    data[..., :3] = 0.0
    data[16:48, 12:52, :3] = 180.0
    data[24:40, 24:40, :3] = 0.0
    data[..., 111] = 22.0
    return data


def test_rgb_mask_preserves_enclosed_black_object_regions() -> None:
    mask = rgb_black_background_foreground_mask(synthetic_wsg())
    assert mask[20, 20]
    assert mask[32, 32]
    assert not mask[0, 0]


def test_dense_channel111_uses_rgb_mask_before_category_plane() -> None:
    category_map = dense_channel111_category_map(
        synthetic_wsg(),
        category_id=7,
        require_alpha=False,
        allow_rgb_mask_fallback=True,
        allow_category_mask_fallback=True,
    )
    assert category_map[20, 20] == 7
    assert category_map[32, 32] == 7
    assert category_map[0, 0] == -1


def test_rgb_mask_supports_normalized_rgb_touching_border() -> None:
    data = np.full((32, 32, 256), -1.0, dtype=np.float32)
    data[..., :3] = 0.0
    data[5:27, :14, :3] = (0.12, 0.45, 0.8)

    mask = rgb_black_background_foreground_mask(data)

    assert mask[10, 1]
    assert mask[10, 10]
    assert not mask[10, 20]


def test_rgb_mask_rejects_a_full_foreground_crop() -> None:
    data = np.full((32, 32, 256), -1.0, dtype=np.float32)
    data[..., :3] = 162.0

    try:
        rgb_black_background_foreground_mask(data)
    except ValueError as error:
        assert "coverage" in str(error)
    else:
        raise AssertionError("A full-foreground crop must be rejected")
