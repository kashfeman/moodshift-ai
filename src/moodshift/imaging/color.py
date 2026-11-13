"""Explicit, independent RGB arrays and float32 CIE LAB conversion."""

import cv2
import numpy as np
from PIL import Image

from moodshift.imaging.exceptions import InvalidImageError
from moodshift.imaging.validation import LabArray, RGBArray, validate_rgb_array


def pil_to_rgb_array(image: Image.Image) -> RGBArray:
    """Copy an RGB PIL image to H x W x 3 uint8; require explicit normalization."""
    if image.mode != "RGB":
        raise InvalidImageError("Normalize the PIL image to RGB first")
    array = np.array(image, dtype=np.uint8, copy=True)
    validate_rgb_array(array)
    return array


def rgb_array_to_pil(array: RGBArray) -> Image.Image:
    """Copy validated uint8 RGB pixels into an independent PIL image."""
    validate_rgb_array(array)
    return Image.fromarray(np.ascontiguousarray(array)).copy()


def rgb_to_lab(array: RGBArray) -> LabArray:
    """Convert RGB to float32 CIE LAB (L 0..100, signed a/b, D65 white)."""
    validate_rgb_array(array)
    return cv2.cvtColor(array.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)


def lab_to_rgb(array: LabArray) -> RGBArray:
    """Convert finite float32 LAB to uint8 RGB, clipping out-of-gamut colors."""
    if (
        not isinstance(array, np.ndarray)
        or array.dtype != np.float32
        or array.ndim != 3
        or array.shape[2] != 3
        or array.shape[0] == 0
        or array.shape[1] == 0
        or not np.isfinite(array).all()
        or np.any((array[..., 0] < 0) | (array[..., 0] > 100))
    ):
        raise InvalidImageError(
            "Expected finite H x W x 3 float32 LAB with L in 0..100"
        )
    rgb = cv2.cvtColor(np.ascontiguousarray(array), cv2.COLOR_LAB2RGB)
    return np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
