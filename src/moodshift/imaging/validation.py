"""Pure validation for image limits and RGB array contracts."""

import numpy as np
from numpy.typing import NDArray

from moodshift.config import AppConfig
from moodshift.imaging.exceptions import ImageLimitError, InvalidImageError

type RGBArray = NDArray[np.uint8]
type LabArray = NDArray[np.float32]


def validate_dimensions(size: tuple[int, int], config: AppConfig) -> None:
    """Check inclusive dimension and pixel limits before allocating pixels."""
    width, height = size
    if width <= 0 or height <= 0:
        raise InvalidImageError("Image dimensions must be positive")
    if (
        width > config.max_image_width
        or height > config.max_image_height
        or width * height > config.max_image_pixels
    ):
        raise ImageLimitError("Image dimensions or pixel count exceed limits")


def validate_rgb_array(array: RGBArray) -> None:
    """Require a nonempty H x W x 3 uint8 array in RGB channel order."""
    if (
        not isinstance(array, np.ndarray)
        or array.dtype != np.uint8
        or array.ndim != 3
        or array.shape[2] != 3
        or array.shape[0] == 0
        or array.shape[1] == 0
    ):
        raise InvalidImageError("Expected a nonempty H x W x 3 uint8 RGB array")
