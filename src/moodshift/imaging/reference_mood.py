"""Conservative, reference-guided photographic color transfer in CIELAB."""

import math

import cv2
import numpy as np
from PIL import Image

from moodshift.imaging.color import lab_to_rgb, pil_to_rgb_array, rgb_to_lab


def _robust_stats(lab: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Use trimmed channel statistics on a bounded-size image sample."""
    height, width = lab.shape[:2]
    factor = min(1.0, 512.0 / max(height, width))
    if factor < 1:
        lab = cv2.resize(
            lab,
            (max(1, round(width * factor)), max(1, round(height * factor))),
            interpolation=cv2.INTER_AREA,
        )
    pixels = lab.reshape(-1, 3)
    low, high = np.percentile(pixels, [5, 95], axis=0)
    trimmed = np.clip(pixels, low, high)
    return trimmed.mean(axis=0), trimmed.std(axis=0)


def reference_mood_boost(
    image: Image.Image, reference: Image.Image, strength: float
) -> Image.Image:
    """Blend bounded LAB statistics while retaining luminance detail."""
    if (
        isinstance(strength, bool)
        or not isinstance(strength, (int, float))
        or not math.isfinite(strength)
        or not 0 <= strength <= 100
    ):
        raise ValueError(
            "Reference Mood Boost strength must be finite and between 0 and 100"
        )
    source = pil_to_rgb_array(image)
    ref = pil_to_rgb_array(reference)
    if strength == 0:
        return image.copy()
    lab = rgb_to_lab(source)
    ref_lab = rgb_to_lab(ref)
    mean, std = _robust_stats(lab)
    ref_mean, ref_std = _robust_stats(ref_lab)
    ratio = np.clip(ref_std / np.maximum(std, 5.0), 0.75, 1.3)
    # Preserve spatial luminance gradients; apply only a modest exposure offset.
    shift = np.clip(ref_mean - mean, [-7, -22, -22], [7, 22, 22])
    adjusted = lab.copy()
    adjusted[..., 0] = np.clip(lab[..., 0] + shift[0] * 0.25, 0, 100)
    adjusted[..., 1:] = (lab[..., 1:] - mean[1:]) * ratio[1:] + mean[1:] + shift[1:]
    # Limit unusually strong chroma and soften the change on likely skin colors.
    chroma = np.linalg.norm(adjusted[..., 1:], axis=2, keepdims=True)
    adjusted[..., 1:] *= np.minimum(1.0, 90.0 / np.maximum(chroma, 1e-6))
    a, b, light = lab[..., 1], lab[..., 2], lab[..., 0]
    skin = (
        (a > 5) & (a < 30) & (b > 8) & (b < 40) & (light > 25) & (light < 90)
    ).astype(np.float32)
    skin = cv2.GaussianBlur(skin, (0, 0), sigmaX=2) if min(skin.shape) > 1 else skin
    amount = (strength / 100.0) * (1.0 - 0.5 * skin[..., None])
    mixed = lab + (adjusted - lab) * amount
    mixed[..., 0] = np.clip(mixed[..., 0], 0, 100)
    return Image.fromarray(lab_to_rgb(np.asarray(mixed, dtype=np.float32)))
