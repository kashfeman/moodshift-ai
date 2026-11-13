"""Smooth, partitioned luminance masks and continuous photographic tone curves."""

import numpy as np
from numpy.typing import NDArray

from moodshift.imaging.grading_parameters import GradeParameters

type FloatArray = NDArray[np.float32]


def smoothstep(value: FloatArray, low: float, high: float) -> FloatArray:
    """Interpolate with zero endpoint slope between two increasing thresholds."""
    if high <= low:
        raise ValueError("smoothstep thresholds must increase")
    fraction = np.clip((value - low) / (high - low), 0, 1)
    return fraction * fraction * (3 - 2 * fraction)


def tonal_masks(lightness: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Partition normalized lightness into smooth shadow/midtone/highlight masks."""
    shadows = 1 - smoothstep(lightness, 0.1, 0.55)
    highlights = smoothstep(lightness, 0.45, 0.9)
    midtones = np.maximum(1 - shadows - highlights, 0)
    return shadows, midtones, highlights


def tone_curve(lightness: FloatArray, parameters: GradeParameters) -> FloatArray:
    """Apply endpoint-safe sigmoid contrast, gamma, and lifted blacks/soft whites."""
    value = np.clip(lightness, 0, 1)
    lower = np.power(value, parameters.contrast)
    upper = np.power(1 - value, parameters.contrast)
    value = lower / (lower + upper)
    value = np.power(value, 1 / parameters.gamma)
    return parameters.black_lift + value * (
        parameters.white_point - parameters.black_lift
    )
