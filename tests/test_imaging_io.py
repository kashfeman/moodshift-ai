"""Generated fixtures exercise bounded local decoding and safe export."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from moodshift.config import AppConfig
from moodshift.imaging.exceptions import (
    ImageExportError,
    ImageLimitError,
    InvalidImageError,
    UnsupportedImageError,
)
from moodshift.imaging.io import export_image, load_image


@pytest.mark.parametrize("format_name", ["PNG", "JPEG", "WEBP"])
def test_load_supported(tmp_path: Path, format_name: str) -> None:
    """Content detection ignores extensions and leaves source bytes untouched."""
    path = tmp_path / "input.data"
    Image.new("RGB", (12, 8), (200, 30, 40)).save(path, format=format_name)
    original = path.read_bytes()
    image = load_image(path)
    assert image.mode == "RGB"
    assert image.size == (12, 8)
    assert np.max(np.abs(np.asarray(image).astype(int) - [200, 30, 40])) <= 5
    image.putpixel((0, 0), (0, 0, 0))
    assert path.read_bytes() == original


@pytest.mark.parametrize("mode", ["L", "LA", "RGBA", "P", "1"])
def test_load_modes(tmp_path: Path, mode: str) -> None:
    """Grayscale, palette, and transparent input become RGB."""
    path = tmp_path / "input.png"
    image = Image.new(mode, (3, 2))
    if mode == "P":
        image.info["transparency"] = 0
    image.save(path)
    loaded = load_image(path)
    assert loaded.mode == "RGB"
    expected = (255, 255, 255) if mode in {"LA", "RGBA", "P"} else (0, 0, 0)
    assert loaded.getpixel((0, 0)) == expected


@pytest.mark.parametrize(
    "kind", ["missing", "empty", "corrupt", "directory", "truncated"]
)
def test_invalid_files(tmp_path: Path, kind: str) -> None:
    """Bad paths and malformed bytes produce domain errors."""
    path = tmp_path / "bad.png"
    if kind == "empty":
        path.touch()
    elif kind == "corrupt":
        path.write_bytes(b"not an image")
    elif kind == "directory":
        path.mkdir()
    elif kind == "truncated":
        Image.new("RGB", (20, 20)).save(path)
        path.write_bytes(path.read_bytes()[:45])
    with pytest.raises((InvalidImageError, UnsupportedImageError)):
        load_image(path)


def test_unsupported_and_animated(tmp_path: Path) -> None:
    """Reject recognized unsupported formats and multi-frame PNGs."""
    path = tmp_path / "input.bmp"
    Image.new("RGB", (4, 4)).save(path)
    with pytest.raises(UnsupportedImageError):
        load_image(path)
    animated = tmp_path / "animated.png"
    Image.new("RGB", (4, 4), "red").save(
        animated, save_all=True, append_images=[Image.new("RGB", (4, 4), "blue")]
    )
    with pytest.raises(UnsupportedImageError, match="multi-frame"):
        load_image(animated)


@pytest.mark.parametrize(
    "settings",
    [
        {"max_upload_bytes": 1},
        {"max_image_width": 7},
        {"max_image_height": 3},
        {"max_image_pixels": 31},
    ],
)
def test_limits(tmp_path: Path, settings: dict[str, int]) -> None:
    """All configured limits are independently enforced."""
    path = tmp_path / "input.png"
    Image.new("RGB", (8, 4)).save(path)
    with pytest.raises(ImageLimitError):
        load_image(path, AppConfig(**settings))


def test_limits_inclusive(tmp_path: Path) -> None:
    """Exact byte, dimension, and pixel boundaries are accepted."""
    path = tmp_path / "input.png"
    Image.new("RGB", (8, 4)).save(path)
    config = AppConfig(
        max_upload_bytes=path.stat().st_size,
        max_image_width=8,
        max_image_height=4,
        max_image_pixels=32,
    )
    assert load_image(path, config).size == (8, 4)


def test_reject_dimensions_before_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An oversized header is rejected before Pillow allocates decoded pixels."""
    from PIL import PngImagePlugin

    path = tmp_path / "input.png"
    Image.new("RGB", (8, 4)).save(path)

    def unexpected_decode(*args: object, **kwargs: object) -> None:
        raise AssertionError("Over-limit images must not be decoded")

    monkeypatch.setattr(PngImagePlugin.PngImageFile, "load", unexpected_decode)
    with pytest.raises(ImageLimitError):
        load_image(path, AppConfig(max_image_width=7))


def test_corrupt_png_checksum(tmp_path: Path) -> None:
    """A damaged PNG chunk is rejected even when the signature is valid."""
    path = tmp_path / "corrupt.png"
    Image.new("RGB", (8, 4)).save(path)
    data = bytearray(path.read_bytes())
    offset = data.index(b"IDAT") + 4
    data[offset] ^= 1
    path.write_bytes(data)
    with pytest.raises(InvalidImageError):
        load_image(path)


@pytest.mark.parametrize("pixel_limit", [20, 10])
def test_pillow_bomb_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pixel_limit: int
) -> None:
    """Both Pillow warning and error paths become ImageLimitError."""
    path = tmp_path / "input.png"
    Image.new("RGB", (8, 4)).save(path)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", pixel_limit)
    with pytest.raises(ImageLimitError, match="bomb"):
        load_image(path)
    assert Image.MAX_IMAGE_PIXELS == pixel_limit


def test_exif_orientation(tmp_path: Path) -> None:
    """Orientation is applied exactly once and swapped dimensions are checked."""
    path = tmp_path / "oriented.png"
    source = Image.new("RGB", (6, 3), "black")
    source.putpixel((0, 0), (255, 0, 0))
    exif = Image.Exif()
    exif[274] = 6
    source.save(path, exif=exif)
    result = load_image(path)
    assert result.size == (3, 6)
    assert result.getpixel((2, 0)) == (255, 0, 0)
    assert 274 not in result.getexif()
    with pytest.raises(ImageLimitError):
        load_image(path, AppConfig(max_image_height=4))


@pytest.mark.parametrize("suffix", [".png", ".jpg", ".jpeg"])
def test_export_roundtrip(tmp_path: Path, suffix: str) -> None:
    """Exported pixels reload with realistic JPEG tolerance and no metadata."""
    image = Image.new("RGBA", (10, 6), (255, 0, 0, 128))
    image.info["comment"] = b"private"
    path = tmp_path / f"output{suffix}"
    assert export_image(image, path) == path
    loaded = load_image(path)
    assert loaded.size == image.size
    assert np.max(np.abs(np.asarray(loaded).astype(int) - [255, 127, 127])) <= 3
    if suffix == ".png":
        assert np.all(np.asarray(loaded) == [255, 127, 127])
    with Image.open(path) as encoded:
        assert "comment" not in encoded.info
        assert not encoded.getexif()
    assert image.mode == "RGBA"
    assert image.info["comment"] == b"private"


def test_export_protection(tmp_path: Path) -> None:
    """Existing source and output paths cannot be overwritten."""
    path = tmp_path / "original.png"
    image = Image.new("RGB", (2, 2))
    image.save(path)
    before = path.read_bytes()
    with pytest.raises(ImageExportError):
        export_image(image, path)
    assert path.read_bytes() == before
    with pytest.raises(ImageExportError):
        export_image(image, tmp_path / "missing" / "out.png")
    with pytest.raises(UnsupportedImageError):
        export_image(image, tmp_path / "out.webp")


@pytest.mark.parametrize("quality", [0, 101, True, 90.5])
def test_export_quality_validation(tmp_path: Path, quality: object) -> None:
    """Reject invalid quality without creating output files."""
    path = tmp_path / "out.jpg"
    with pytest.raises(ImageExportError):
        export_image(Image.new("RGB", (2, 2)), path, jpeg_quality=quality)
    assert not path.exists()
