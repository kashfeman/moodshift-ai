"""Verify actual detail preservation and neural residual influence independently."""

import numpy as np
import pytest
from PIL import Image

from moodshift.imaging.exceptions import InvalidImageError
from moodshift.imaging.reconstruction import reconstruct_hybrid, reconstruct_neural


def test_identity_and_original_unchanged() -> None:
    """An unchanged neural result must preserve every original pixel."""
    pixels = np.random.default_rng(1).integers(0, 256, (64, 80, 3), dtype=np.uint8)
    content = Image.fromarray(pixels)
    reduced = content.resize((20, 16), Image.Resampling.LANCZOS)
    result = reconstruct_hybrid(content, reduced)
    np.testing.assert_array_equal(result, pixels)
    np.testing.assert_array_equal(content, pixels)
    assert result is not content


def test_constant_neural_color_shift_retains_fine_detail() -> None:
    """Real transferred color remains while source checkerboard detail survives."""
    grid = (np.indices((64, 64)).sum(axis=0) % 2 * 40 + 90).astype(np.uint8)
    pixels = np.repeat(grid[..., None], 3, axis=2)
    content = Image.fromarray(pixels)
    base = np.array(content.resize((16, 16), Image.Resampling.LANCZOS))
    neural = Image.fromarray(base + np.array([15, 5, 0], dtype=np.uint8))
    result = np.array(reconstruct_hybrid(content, neural)).astype(int)
    np.testing.assert_allclose(
        result - pixels, np.broadcast_to([15, 5, 0], result.shape)
    )


def test_generated_texture_is_attenuated_without_blurring_source() -> None:
    """Uncorrelated neural checkerboard is suppressed on a constant source."""
    content = Image.new("RGB", (64, 64), (128, 128, 128))
    noise = (np.indices((16, 16)).sum(axis=0) % 2 * 80 + 88).astype(np.uint8)
    neural = Image.fromarray(np.repeat(noise[..., None], 3, axis=2))
    result = np.asarray(reconstruct_hybrid(content, neural), dtype=float)
    assert result.std() < 2
    assert result.mean() == pytest.approx(128, abs=1)


def test_requires_explicit_rgb_normalization() -> None:
    """Invalid modes use the existing typed image exception."""
    with pytest.raises(InvalidImageError):
        reconstruct_hybrid(Image.new("L", (20, 20)), Image.new("RGB", (10, 10)))


def test_neural_reconstruction_preserves_detail_and_uses_neural_color() -> None:
    """The result retains high-frequency detail and changes with neural output."""
    grid = (np.indices((64, 64)).sum(axis=0) % 2 * 50 + 90).astype(np.uint8)
    content = Image.fromarray(np.repeat(grid[..., None], 3, axis=2))
    base = np.asarray(content.resize((16, 16), Image.Resampling.LANCZOS)).copy()
    warm = Image.fromarray(
        np.clip(base.astype(int) + [20, 4, 0], 0, 255).astype("uint8")
    )
    cool = Image.fromarray(
        np.clip(base.astype(int) + [0, 4, 20], 0, 255).astype("uint8")
    )
    result = np.asarray(reconstruct_neural(content, warm)).astype(float)
    other = np.asarray(reconstruct_neural(content, cool)).astype(float)
    direct = np.asarray(warm.resize(content.size, Image.Resampling.LANCZOS)).astype(
        float
    )
    assert (
        np.abs(np.diff(result, axis=1)).mean()
        > np.abs(np.diff(direct, axis=1)).mean() + 10
    )
    assert np.abs(result - other).mean() > 5
    assert result.shape == (64, 64, 3)
