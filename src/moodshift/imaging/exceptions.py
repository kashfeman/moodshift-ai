"""Domain errors exposed by the imaging boundary."""


class ImageError(ValueError):
    """Base error for invalid images and imaging operations."""


class InvalidImageError(ImageError):
    """An image is missing, unreadable, corrupt, or has invalid data."""


class UnsupportedImageError(ImageError):
    """The encoded image format or image mode is unsupported."""


class ImageLimitError(InvalidImageError):
    """Encoded size, dimensions, or decoded pixels exceed configured limits."""


class ImageExportError(ImageError):
    """An output cannot be written safely to the requested destination."""
