"""Photographic grade invariants and numerical behavior on generated images."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from PIL import Image

from moodshift.imaging.color import pil_to_rgb_array, rgb_to_lab
from moodshift.imaging.effects import film_grain, highlight_glow
from moodshift.imaging.grading import adjust_lab_chroma, apply_grade, blend_images
from moodshift.imaging.grading_parameters import GradeParameters
from moodshift.imaging.reference_images import (
    synthetic_scene,
    synthetic_style_reference,
)
from moodshift.imaging.tonal import tonal_masks, tone_curve
from moodshift.presets import PRESETS, MoodPreset, get_preset


@pytest.mark.parametrize("preset", PRESETS, ids=lambda p: p.id)
def test_presets_affect_pixels_and_preserve_input(preset: MoodPreset) -> None:
    """Each full grade changes pixels, preserves shape, and leaves the source alone."""
    image = synthetic_scene()
    before = image.tobytes()
    result = apply_grade(image, preset.grading)
    assert result.size == image.size and result.mode == "RGB"
    array = np.asarray(result)
    assert array.dtype == np.uint8 and np.isfinite(array).all()
    assert np.abs(array.astype(float) - np.asarray(image)).mean() > 1
    assert image.tobytes() == before
    zero = apply_grade(image, preset.grading, strength=0)
    assert zero.tobytes() == before


def test_distinct_grades_and_synthetic_references() -> None:
    """Presets and their deterministic synthetic references are pairwise distinct."""
    image = synthetic_scene()
    outputs = [
        np.asarray(apply_grade(image, preset.grading)).astype(float)
        for preset in PRESETS
    ]
    for index, first in enumerate(outputs):
        for second in outputs[index + 1 :]:
            assert np.abs(first - second).mean() > 2
    references = [synthetic_style_reference(p.grading) for p in PRESETS]
    assert len({image.tobytes() for image in references}) == 5
    assert all(image.size == (128, 128) for image in references)
    assert (
        references[0].tobytes()
        == synthetic_style_reference(PRESETS[0].grading).tobytes()
    )


def test_selective_warm_cool_and_lab_luminance() -> None:
    """CIELAB chroma adjustments preserve L and implement intended color directions."""
    gray = Image.fromarray(
        np.repeat(np.array([[48, 128, 216]], np.uint8)[..., None], 3, axis=-1)
    )
    lab = rgb_to_lab(pil_to_rgb_array(gray))
    warm = adjust_lab_chroma(lab, get_preset("sunset").grading)
    cool = adjust_lab_chroma(lab, get_preset("cold-blue").grading)
    assert np.array_equal(warm[..., 0], lab[..., 0])
    assert np.array_equal(cool[..., 0], lab[..., 0])
    assert np.all(warm[..., 2] > lab[..., 2])
    assert np.all(cool[..., 2] < lab[..., 2])
    cinematic = adjust_lab_chroma(lab, get_preset("cinematic").grading)
    assert cinematic[0, 0, 1] < 0 and cinematic[0, 0, 2] < 0
    assert cinematic[0, 2, 1] > 0 and cinematic[0, 2, 2] > 0
    red = Image.new("RGB", (4, 4), (230, 20, 10))
    output = np.asarray(apply_grade(red, GradeParameters(saturation=0.95)))
    assert np.all(output[..., 0] > output[..., 2])


def test_masks_smooth_and_partitioned() -> None:
    """Masks sum to one and have smooth monotonic shadow/highlight transitions."""
    ramp = np.linspace(0, 1, 2001, dtype=np.float32)
    shadows, midtones, highlights = tonal_masks(ramp)
    assert np.allclose(shadows + midtones + highlights, 1)
    for mask in (shadows, midtones, highlights):
        assert mask.min() >= 0 and mask.max() <= 1
        assert np.abs(np.diff(mask)).max() < 0.003
    assert np.all(np.diff(shadows) <= 0)
    assert np.all(np.diff(highlights) >= 0)
    assert shadows[0] == 1 and highlights[-1] == 1


def test_tone_curves_and_grain() -> None:
    """Tone curves remain monotonic and grain is deterministic and monochromatic."""
    ramp = np.linspace(0, 1, 1000, dtype=np.float32)
    for preset in PRESETS:
        values = tone_curve(ramp, preset.grading)
        assert np.isfinite(values).all() and np.all(np.diff(values) >= 0)
        assert 0 <= values.min() <= values.max() <= 1
    pixels = np.full((30, 40, 3), 0.5, dtype=np.float32)
    before = pixels.copy()
    first = film_grain(pixels, 0.006, 7)
    assert np.array_equal(first, film_grain(pixels, 0.006, 7))
    assert not np.array_equal(first, film_grain(pixels, 0.006, 8))
    assert np.array_equal(first[..., 0], first[..., 2])
    assert np.array_equal(pixels, before)
    vintage = get_preset("vintage").grading
    image = Image.new("RGB", (40, 30), (128, 128, 128))
    assert (
        apply_grade(image, vintage, grain_enabled=False).tobytes()
        != apply_grade(image, vintage).tobytes()
    )


def test_glow_is_highlight_controlled() -> None:
    """Dark images receive no glow; a bright point creates a bounded nearby halo."""
    dark = np.full((41, 41, 3), 0.2, dtype=np.float32)
    assert np.array_equal(dark, highlight_glow(dark, 0.2))
    dark[20, 20] = 1
    result = highlight_glow(dark, 0.2)
    assert result[20, 21, 0] > dark[20, 21, 0]
    assert np.all(result >= dark) and result.max() <= 1


def test_strength_interpolation() -> None:
    """Half strength interpolates the full appearance within one code value."""
    source = synthetic_scene()
    parameters = get_preset("sunset").grading
    full = np.asarray(apply_grade(source, parameters)).astype(float)
    halfway = np.asarray(apply_grade(source, parameters, strength=0.5)).astype(float)
    assert (
        np.max(np.abs(halfway - (np.asarray(source).astype(float) + full) / 2)) <= 0.5
    )
    for value in (-1, 2, float("nan"), True):
        with pytest.raises(ValueError):
            blend_images(source, source, value)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("contrast", 2),
        ("gamma", 0),
        ("saturation", float("nan")),
        ("vibrance", 1),
        ("grain", -1),
        ("bloom", True),
        ("shadows", [1, 2]),
        ("midtones", (1,)),
        ("highlights", (0, float("inf"))),
    ],
)
def test_invalid_grading_parameters(name: str, value: object) -> None:
    """Processing parameters are immutable and constrain extreme effects."""
    with pytest.raises(ValueError, match=name):
        GradeParameters(**{name: value})
    with pytest.raises(FrozenInstanceError):
        PRESETS[0].grading.saturation = 2
