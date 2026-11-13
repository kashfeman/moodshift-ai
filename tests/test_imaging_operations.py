"""Numerical contracts for normalization, resize, color, and statistics."""

import numpy as np
import pytest
from PIL import Image

from moodshift.imaging.color import (
    lab_to_rgb,
    pil_to_rgb_array,
    rgb_array_to_pil,
    rgb_to_lab,
)
from moodshift.imaging.exceptions import InvalidImageError, UnsupportedImageError
from moodshift.imaging.statistics import image_statistics
from moodshift.imaging.transforms import resize_to_fit, to_rgb


def test_alpha_and_independence() -> None:
    """Alpha composites over white without mutating original pixels or metadata."""
    source = Image.new("RGBA", (2, 1), (0, 0, 255, 128))
    result = to_rgb(source)
    assert result.getpixel((0, 0)) == (127, 127, 255)
    assert source.getpixel((0, 0)) == (0, 0, 255, 128)
    gray = to_rgb(Image.new("L", (1, 1), 42))
    assert gray.getpixel((0, 0)) == (42, 42, 42)
    assert to_rgb(Image.new("CMYK", (1, 1), (0, 255, 255, 0))).getpixel((0, 0)) == (
        255,
        0,
        0,
    )
    with pytest.raises(UnsupportedImageError):
        to_rgb(Image.new("I", (1, 1)))


@pytest.mark.parametrize(
    ("size", "bounds", "expected"),
    [
        ((100, 50), (40, 40), (40, 20)),
        ((50, 100), (40, 40), (20, 40)),
        ((10, 5), (40, 40), (10, 5)),
        ((101, 31), (50, 50), (50, 15)),
    ],
)
def test_resize(
    size: tuple[int, int], bounds: tuple[int, int], expected: tuple[int, int]
) -> None:
    """Resize fits bounds, preserves aspect ratio, and never upscales."""
    source = Image.new("RGB", size)
    result = resize_to_fit(source, bounds)
    assert result.size == expected
    assert source.size == size
    assert result is not source
    assert abs(result.height - result.width * size[1] / size[0]) <= 1


@pytest.mark.parametrize("bounds", [(0, 10), (-1, 10), (True, 10), (1.5, 10)])
def test_invalid_resize(bounds: tuple[int, int]) -> None:
    """Invalid resizing bounds fail explicitly."""
    with pytest.raises(InvalidImageError):
        resize_to_fit(Image.new("RGB", (2, 2)), bounds)


def test_rgb_array_channel_order_and_ownership() -> None:
    """RGB channels stay ordered and all conversion results own their pixels."""
    source = Image.new("RGB", (2, 1), (255, 20, 10))
    array = pil_to_rgb_array(source)
    assert array.dtype == np.uint8
    assert array.shape == (1, 2, 3)
    assert array[0, 0].tolist() == [255, 20, 10]
    image = rgb_array_to_pil(array[:, ::-1])
    array[0, 0] = 0
    assert image.getpixel((1, 0)) == (255, 20, 10)
    assert source.getpixel((0, 0)) == (255, 20, 10)
    with pytest.raises(InvalidImageError):
        pil_to_rgb_array(Image.new("L", (2, 2)))


@pytest.mark.parametrize(
    "array",
    [
        np.zeros((2, 2)),
        np.zeros((2, 2, 3)),
        np.zeros((2, 2, 4), dtype=np.uint8),
        np.zeros((0, 2, 3), dtype=np.uint8),
        np.zeros((2, 2, 3), dtype=np.int16),
    ],
)
def test_invalid_rgb_arrays(array: np.ndarray) -> None:
    """Invalid dtypes and shapes never silently scale, wrap, or reorder values."""
    for operation in (rgb_array_to_pil, rgb_to_lab, image_statistics):
        with pytest.raises(InvalidImageError):
            operation(array)


def test_lab_roundtrip() -> None:
    """Float LAB preserves RGB within OpenCV quantization tolerance."""
    array = np.random.default_rng(42).integers(0, 256, (30, 40, 3), dtype=np.uint8)
    restored = lab_to_rgb(rgb_to_lab(array))
    assert np.max(np.abs(restored.astype(int) - array.astype(int))) <= 2
    primaries = np.array([[[255, 0, 0], [0, 0, 255], [255, 255, 255]]], dtype=np.uint8)
    lab = rgb_to_lab(primaries)
    assert lab.dtype == np.float32
    assert lab[0, 0, 0] == pytest.approx(53.24, abs=0.2)
    assert lab[0, 1, 0] == pytest.approx(32.30, abs=0.2)
    assert lab[0, 2, 0] == pytest.approx(100, abs=0.1)


@pytest.mark.parametrize(
    "array",
    [
        np.zeros((2, 2, 3)),
        np.zeros((2, 2), dtype=np.float32),
        np.full((1, 1, 3), np.nan, dtype=np.float32),
        np.full((1, 1, 3), np.inf, dtype=np.float32),
        np.full((1, 1, 3), 101, dtype=np.float32),
    ],
)
def test_invalid_lab_arrays(array: np.ndarray) -> None:
    """LAB input must be finite float32 with valid shape and lightness."""
    with pytest.raises(InvalidImageError):
        lab_to_rgb(array)


def test_statistics() -> None:
    """Known patches establish brightness, contrast, channel means, and scale."""
    array = np.array([[[0, 0, 0], [255, 255, 255]]], dtype=np.uint8)
    before = array.copy()
    stats = image_statistics(array)
    assert stats.brightness == pytest.approx(0.5)
    assert stats.contrast == pytest.approx(0.5)
    assert stats.rgb_mean == pytest.approx((0.5, 0.5, 0.5))
    assert stats.rgb_std == pytest.approx((0.5, 0.5, 0.5))
    assert stats.lab_mean == pytest.approx((50, 0, 0), abs=0.1)
    assert stats.lab_std == pytest.approx((50, 0, 0), abs=0.1)
    assert np.array_equal(array, before)
    red = image_statistics(np.array([[[255, 0, 0]]], dtype=np.uint8))
    assert red.brightness == pytest.approx(0.2126)
    assert red.contrast == 0
