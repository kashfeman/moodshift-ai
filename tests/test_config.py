"""Tests for configuration defaults, invariants, and immutability."""

from dataclasses import FrozenInstanceError

import pytest

from moodshift.config import DEFAULT_CONFIG, AppConfig
from moodshift.presets import get_preset


def test_defaults() -> None:
    """Defaults provide bounded inputs and a registered preset."""
    assert DEFAULT_CONFIG == AppConfig()
    assert DEFAULT_CONFIG.max_upload_bytes == 10 * 1024 * 1024
    assert DEFAULT_CONFIG.max_image_width == 4096
    assert DEFAULT_CONFIG.max_image_height == 4096
    assert DEFAULT_CONFIG.max_image_pixels == 16_000_000
    assert DEFAULT_CONFIG.default_strength == 0.5
    assert DEFAULT_CONFIG.output_format == "PNG"
    assert get_preset(DEFAULT_CONFIG.default_preset_id).name == "Cinematic"


@pytest.mark.parametrize(
    "field",
    ["max_upload_bytes", "max_image_width", "max_image_height", "max_image_pixels"],
)
@pytest.mark.parametrize("value", [0, -1, True, 1.5, "1024", None])
def test_invalid_limits(field: str, value: object) -> None:
    """Limits require positive integers, excluding booleans."""
    with pytest.raises(ValueError, match=field):
        AppConfig(**{field: value})


@pytest.mark.parametrize(
    "strength", [-0.1, 1.1, float("nan"), float("inf"), True, "0.5"]
)
def test_invalid_strength(strength: object) -> None:
    """Reject out-of-range and nonnumeric strength values."""
    with pytest.raises(ValueError, match="default_strength"):
        AppConfig(default_strength=strength)


@pytest.mark.parametrize("strength", [0.0, 1.0])
def test_strength_boundaries(strength: float) -> None:
    """Both strength endpoints are valid."""
    assert AppConfig(default_strength=strength).default_strength == strength


@pytest.mark.parametrize("preset_id", ["", "   ", None])
def test_empty_default_preset(preset_id: object) -> None:
    """A preset ID must be meaningful text."""
    with pytest.raises(ValueError, match="default_preset_id"):
        AppConfig(default_preset_id=preset_id)


def test_output_format() -> None:
    """Support only the declared output formats."""
    assert AppConfig(output_format="JPEG").output_format == "JPEG"
    with pytest.raises(ValueError, match="output_format"):
        AppConfig(output_format="GIF")


def test_config_is_frozen() -> None:
    """Shared default settings cannot be mutated."""
    with pytest.raises(FrozenInstanceError):
        DEFAULT_CONFIG.default_strength = 0.9
