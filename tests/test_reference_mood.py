"""Reference color transfer preserves details and remains an optional UI stage."""

import numpy as np
import pytest
from PIL import Image

from moodshift.imaging.reference_mood import reference_mood_boost


def test_reference_boost_strength_and_direction():
    """Zero is exact, and two references produce distinct bounded RGB images."""
    rng = np.random.default_rng(5)
    content = Image.fromarray(rng.integers(40, 210, (72, 96, 3), dtype=np.uint8))
    warm = Image.new("RGB", (48, 48), (235, 125, 60))
    cool = Image.new("RGB", (48, 48), (65, 110, 205))
    assert reference_mood_boost(content, warm, 0).tobytes() == content.tobytes()
    a = np.asarray(reference_mood_boost(content, warm, 100), dtype=np.float32)
    b = np.asarray(reference_mood_boost(content, cool, 100), dtype=np.float32)
    assert a.shape == b.shape == (72, 96, 3)
    assert np.abs(a - b).mean() > 5
    assert (a[..., 0] - a[..., 2]).mean() > (b[..., 0] - b[..., 2]).mean()
    for value in (-1, 101, float("nan"), True):
        with pytest.raises(ValueError):
            reference_mood_boost(content, warm, value)


def test_flat_and_tiny_images_remain_finite():
    """Constant references and tiny images do not divide by zero."""
    image = Image.new("RGB", (1, 1), (50, 100, 150))
    assert reference_mood_boost(image, image, 100).size == (1, 1)
