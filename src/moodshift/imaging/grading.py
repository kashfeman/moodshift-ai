"""CIELAB-based photographic grading, with explicit RGB and strength contracts."""

import math

import numpy as np
from PIL import Image

from moodshift.imaging.color import lab_to_rgb, pil_to_rgb_array, rgb_to_lab
from moodshift.imaging.effects import film_grain, highlight_glow
from moodshift.imaging.grading_parameters import GradeParameters
from moodshift.imaging.tonal import tonal_masks, tone_curve
from moodshift.imaging.validation import LabArray


def adjust_lab_chroma(lab: LabArray, parameters: GradeParameters) -> LabArray:
    """Grade a/b while leaving L untouched; taper offsets near black and white.

    Vibrance favors low-chroma colors and reduces its boost for warm hues.
    Conversion back to RGB may clip out-of-gamut colors; offsets stay modest.
    """
    result = lab.copy()
    lightness = lab[..., 0] / 100
    masks = tonal_masks(lightness)
    chroma = np.linalg.norm(lab[..., 1:], axis=-1)
    warm = (lab[..., 1] > 0) & (lab[..., 2] > 0)
    boost = parameters.vibrance * np.square(1 - np.clip(chroma / 80, 0, 1))
    boost = boost * (1 - warm.astype(np.float32) * 0.5)
    result[..., 1:] *= (parameters.saturation * (1 + boost))[..., None]
    envelope = np.sqrt(np.maximum(4 * lightness * (1 - lightness), 0))
    for mask, offset in zip(
        masks,
        (parameters.shadows, parameters.midtones, parameters.highlights),
        strict=True,
    ):
        result[..., 1:] += (mask * envelope)[..., None] * np.array(offset, np.float32)
    return result


def blend_images(
    original: Image.Image, stylized: Image.Image, strength: float
) -> Image.Image:
    """Interpolate encoded RGB pixels once into independent rounded uint8 pixels."""
    if (
        isinstance(strength, bool)
        or not isinstance(strength, (int, float))
        or not math.isfinite(strength)
        or not 0 <= strength <= 1
    ):
        raise ValueError("strength must be finite and in [0,1]")
    if original.size != stylized.size:
        raise ValueError("Blend images must have identical dimensions")
    first = pil_to_rgb_array(original)
    second = pil_to_rgb_array(stylized)
    value = (
        first.astype(np.float32) * (1 - strength) + second.astype(np.float32) * strength
    )
    return Image.fromarray(np.rint(np.clip(value, 0, 255)).astype(np.uint8))


def apply_grade(
    image: Image.Image,
    parameters: GradeParameters,
    *,
    strength: float = 1.0,
    seed: int = 0,
    grain_enabled: bool = True,
) -> Image.Image:
    """Grade normalized RGB pixels without mutating the image or its geometry."""
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 4294967295]")
    if type(grain_enabled) is not bool:
        raise ValueError("grain_enabled must be boolean")
    array = pil_to_rgb_array(image)
    lab = rgb_to_lab(array)
    graded_lab = adjust_lab_chroma(lab, parameters)
    graded_lab[..., 0] = tone_curve(lab[..., 0] / 100, parameters) * 100
    rgb = lab_to_rgb(graded_lab).astype(np.float32) / 255
    rgb = highlight_glow(rgb, parameters.bloom)
    if grain_enabled:
        rgb = film_grain(rgb, parameters.grain, seed)
    graded = Image.fromarray(np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    return blend_images(image, graded, strength)
