"""Tests for the descriptive preset catalog."""

from dataclasses import FrozenInstanceError

import pytest

from moodshift.presets import PRESETS, MoodPreset, get_preset


def test_catalog() -> None:
    """All five requested presets have distinct stable IDs and descriptions."""
    assert isinstance(PRESETS, tuple)
    assert [preset.name for preset in PRESETS] == [
        "Cinematic",
        "Sunset",
        "Cold Blue",
        "Vintage",
        "Dreamy",
    ]
    assert len({preset.id for preset in PRESETS}) == 5
    assert all(preset.description.strip() for preset in PRESETS)


@pytest.mark.parametrize("preset", PRESETS, ids=lambda preset: preset.id)
def test_lookup(preset: MoodPreset) -> None:
    """Every stable ID resolves to its catalog entry."""
    assert get_preset(preset.id) is preset


@pytest.mark.parametrize("preset_id", ["unknown", "", "Cinematic"])
def test_unknown_preset(preset_id: str) -> None:
    """Unknown IDs raise an explicit error without silently falling back."""
    with pytest.raises(KeyError) as error:
        get_preset(preset_id)
    assert error.value.args == (preset_id,)


def test_preset_is_frozen() -> None:
    """Catalog entries cannot be changed after construction."""
    with pytest.raises(FrozenInstanceError):
        PRESETS[0].name = "Changed"


@pytest.mark.parametrize("field", ["id", "name", "description"])
@pytest.mark.parametrize("value", ["", "   ", None])
def test_invalid_metadata(field: str, value: object) -> None:
    """Each metadata field requires non-empty text."""
    values = {"id": "test", "name": "Test", "description": "Test style."}
    values[field] = value
    with pytest.raises(ValueError, match=field):
        MoodPreset(**values)
