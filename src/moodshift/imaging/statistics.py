"""Descriptive color and luminance statistics, independent of inference."""

from dataclasses import dataclass

import numpy as np

from moodshift.imaging.color import rgb_to_lab
from moodshift.imaging.validation import RGBArray, validate_rgb_array


@dataclass(frozen=True, slots=True)
class ImageStatistics:
    """Population statistics: RGB in 0..1, LAB in CIE units.

    Brightness is mean BT.709 luma of gamma-encoded RGB in 0..1; contrast is
    its population standard deviation, not physical luminance or exposure.
    """

    brightness: float
    contrast: float
    rgb_mean: tuple[float, float, float]
    rgb_std: tuple[float, float, float]
    lab_mean: tuple[float, float, float]
    lab_std: tuple[float, float, float]


def image_statistics(array: RGBArray) -> ImageStatistics:
    """Measure global luma and per-channel means and population deviations."""
    validate_rgb_array(array)
    rgb = array.astype(np.float64) / 255.0
    lab = rgb_to_lab(array).astype(np.float64)
    luma = rgb @ np.array([0.2126, 0.7152, 0.0722])
    return ImageStatistics(
        brightness=float(luma.mean()),
        contrast=float(luma.std()),
        rgb_mean=tuple(float(rgb[..., i].mean()) for i in range(3)),
        rgb_std=tuple(float(rgb[..., i].std()) for i in range(3)),
        lab_mean=tuple(float(lab[..., i].mean()) for i in range(3)),
        lab_std=tuple(float(lab[..., i].std()) for i in range(3)),
    )
