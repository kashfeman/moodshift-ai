"""Deterministic image mode normalization and aspect-preserving resizing."""

from PIL import Image

from moodshift.imaging.exceptions import InvalidImageError, UnsupportedImageError


def to_rgb(image: Image.Image) -> Image.Image:
    """Return independent 8-bit RGB pixels, compositing alpha over white.

    Palette transparency and grayscale alpha follow the same rule. CMYK uses
    Pillow's basic conversion; ICC profiles are not interpreted. High-bit-depth
    integer and floating-point modes are rejected rather than silently clipped.
    """
    if image.mode not in {"1", "L", "LA", "P", "RGB", "RGBA", "CMYK", "YCbCr"}:
        raise UnsupportedImageError(f"Unsupported image mode: {image.mode}")
    if image.width <= 0 or image.height <= 0:
        raise InvalidImageError("Image must contain pixels")
    if "A" in image.getbands() or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        result = Image.alpha_composite(background, rgba).convert("RGB")
    else:
        result = image.convert("RGB")
    result.info.clear()
    return result


def resize_to_fit(image: Image.Image, max_size: tuple[int, int]) -> Image.Image:
    """Return a copy fitting the box with LANCZOS; never upscale or stretch.

    Integer dimensions can introduce at most one pixel of rounding error.
    """
    if len(max_size) != 2 or any(type(v) is not int or v <= 0 for v in max_size):
        raise InvalidImageError("Resize bounds must be two positive integers")
    if image.width <= 0 or image.height <= 0:
        raise InvalidImageError("Image must contain pixels")
    result = image.copy()
    result.thumbnail(max_size, Image.Resampling.LANCZOS)
    return result
