"""Alignment and identity tests for trained decoder reconstruction."""

import numpy as np
import pytest
from PIL import Image, ImageOps

from moodshift.imaging.decoder_reconstruction import (
    center_crop_for_decoder,
    decoder_input_from_original,
    reconstruct_decoder_color,
)


@pytest.mark.parametrize("size", [(241, 160), (160, 241), (200, 160), (160, 160)])
def test_center_crop_alignment_and_identity(size):
    """Preserve the exact source crop and existing 96-pixel decoder preprocessing."""
    width, height = size
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    source = Image.fromarray(
        np.stack((x % 256, y % 256, (x + y) % 256), axis=2).astype("uint8")
    )
    crop = center_crop_for_decoder(source)
    native = decoder_input_from_original(source)
    previous = ImageOps.fit(source, (96, 96), method=Image.Resampling.LANCZOS)
    assert native.tobytes() == previous.tobytes()
    assert crop.size == (min(size), min(size))
    crop_low = np.asarray(crop.resize((96, 96), Image.Resampling.LANCZOS)).astype(float)
    assert np.abs(crop_low - np.asarray(native).astype(float)).mean() < 1
    identity = reconstruct_decoder_color(crop, native, native)
    assert identity.tobytes() == crop.tobytes()


def test_color_residual_preserves_detail_and_handles_bad_dimensions():
    """A constant trained-output change transfers without losing source texture."""
    grid = (np.indices((192, 192)).sum(axis=0) % 2 * 40 + 90).astype("uint8")
    crop = Image.fromarray(np.repeat(grid[..., None], 3, axis=2))
    low = decoder_input_from_original(crop)
    before = np.asarray(low).astype(int)
    output = Image.fromarray(np.clip(before + [14, 4, 0], 0, 255).astype("uint8"))
    reconstructed = np.asarray(reconstruct_decoder_color(crop, low, output)).astype(int)
    change = reconstructed - np.asarray(crop).astype(int)
    assert reconstructed.shape == (192, 192, 3)
    assert change[..., 0].mean() > change[..., 2].mean() + 5
    assert np.abs(np.diff(reconstructed.astype(float), axis=1)).mean() > 30
    with pytest.raises(ValueError):
        reconstruct_decoder_color(crop, low, Image.new("RGB", (20, 30)))


def test_bounded_amplification_follows_decoder_change():
    """A higher factor strengthens the same residual; invalid factors fail."""
    crop = Image.new("RGB", (192, 192), (110, 120, 130))
    low = decoder_input_from_original(crop)
    output = Image.new("RGB", (96, 96), (120, 124, 130))
    single = np.asarray(reconstruct_decoder_color(crop, low, output, 1)).astype(float)
    boosted = np.asarray(reconstruct_decoder_color(crop, low, output, 1.5)).astype(
        float
    )
    source = np.asarray(crop).astype(float)
    assert np.abs(boosted - source).mean() > np.abs(single - source).mean()
    assert np.sum((boosted - source) * (single - source)) > 0
    assert reconstruct_decoder_color(crop, low, low, 2).tobytes() == crop.tobytes()
    for invalid in (0, 2.1, float("nan"), True):
        with pytest.raises(ValueError):
            reconstruct_decoder_color(crop, low, output, invalid)
