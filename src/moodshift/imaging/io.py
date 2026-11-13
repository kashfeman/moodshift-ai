"""Bounded local image loading and non-overwriting image export."""

import io
import warnings
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from moodshift.config import DEFAULT_CONFIG, AppConfig
from moodshift.imaging.exceptions import (
    ImageExportError,
    ImageLimitError,
    InvalidImageError,
    UnsupportedImageError,
)
from moodshift.imaging.transforms import to_rgb
from moodshift.imaging.validation import validate_dimensions

SUPPORTED_FORMATS = ("PNG", "JPEG", "WEBP")


def load_image(path: str | Path, config: AppConfig = DEFAULT_CONFIG) -> Image.Image:
    """Load one static PNG/JPEG/WebP as detached, EXIF-oriented RGB pixels.

    Format is detected from content, not extension. Encoded reads are bounded;
    raw dimensions are checked before decoding, oriented sizes before rotation.
    Pillow bomb warnings become errors without changing global safety limits.
    """
    source = Path(path)
    try:
        if not source.is_file():
            raise InvalidImageError("Input must be an existing regular file")
        with source.open("rb") as stream:
            if source.stat().st_size > config.max_upload_bytes:
                raise ImageLimitError("Encoded image exceeds byte limit")
            data = stream.read(config.max_upload_bytes + 1)
        if not data:
            raise InvalidImageError("Image file is empty")
        if len(data) > config.max_upload_bytes:
            raise ImageLimitError("Encoded image exceeds byte limit")
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data), formats=SUPPORTED_FORMATS) as image:
                validate_dimensions(image.size, config)
                if getattr(image, "n_frames", 1) != 1:
                    raise UnsupportedImageError(
                        "Animated or multi-frame images are unsupported"
                    )
                # Verify structure separately; verify() invalidates the decoder.
                image.verify()
            with Image.open(io.BytesIO(data), formats=SUPPORTED_FORMATS) as image:
                # PNG getexif() may decode to find trailing EXIF chunks, so it
                # must follow verification and the raw dimension/pixel checks.
                orientation = image.getexif().get(274, 1)
                oriented_size = (
                    image.size[::-1] if orientation in (5, 6, 7, 8) else image.size
                )
                validate_dimensions(oriented_size, config)
                image.load()
                return to_rgb(ImageOps.exif_transpose(image))
    except (Image.DecompressionBombWarning, Image.DecompressionBombError) as error:
        raise ImageLimitError(
            "Pillow decompression-bomb safety limit exceeded"
        ) from error
    except UnidentifiedImageError as error:
        # Pillow cannot reliably distinguish an unknown format from corrupt bytes.
        raise UnsupportedImageError(
            "Unrecognized, unsupported, or corrupt image"
        ) from error
    except (OSError, SyntaxError, ValueError, EOFError) as error:
        if isinstance(error, (InvalidImageError, UnsupportedImageError)):
            raise
        raise InvalidImageError("Cannot read or decode image") from error


def export_image(
    image: Image.Image, path: str | Path, *, jpeg_quality: int = 95
) -> Path:
    """Create a new PNG/JPEG file; never overwrite an existing path.

    Parent directories must exist. JPEG is lossy, with explicit quality and no
    chroma subsampling. Alpha is composited over white; metadata is stripped.
    The full encoding completes before exclusive destination creation.
    """
    destination = Path(path)
    formats = {".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG"}
    output_format = formats.get(destination.suffix.lower())
    if output_format is None:
        raise UnsupportedImageError("Export supports .png, .jpg, and .jpeg only")
    if type(jpeg_quality) is not int or not 1 <= jpeg_quality <= 100:
        raise ImageExportError("JPEG quality must be an integer between 1 and 100")
    rgb = to_rgb(image)
    try:
        buffer = io.BytesIO()
        options = (
            {"quality": jpeg_quality, "subsampling": 0}
            if output_format == "JPEG"
            else {}
        )
        rgb.save(buffer, format=output_format, **options)
        with destination.open("xb") as stream:
            stream.write(buffer.getbuffer())
    except OSError as error:
        raise ImageExportError(
            "Cannot create output; existing files are never overwritten"
        ) from error
    return destination
