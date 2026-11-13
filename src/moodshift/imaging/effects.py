"""Controlled highlight diffusion and deterministic monochromatic film grain."""

import cv2
import numpy as np

from moodshift.imaging.tonal import FloatArray, smoothstep


def highlight_glow(rgb: FloatArray, amount: float) -> FloatArray:
    """Screen a soft bright-pixel halo over RGB in [0,1], preserving dark detail."""
    if amount == 0:
        return rgb.copy()
    luma = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    mask = smoothstep(luma, 0.55, 0.95)
    bright = rgb * mask[..., None]
    sigma = max(0.6, min(max(rgb.shape[:2]) * 0.012, 12.0))
    halo = cv2.GaussianBlur(
        bright, (0, 0), sigmaX=sigma, borderType=cv2.BORDER_REFLECT_101
    )
    return rgb + (1 - rgb) * halo * amount


def film_grain(rgb: FloatArray, amount: float, seed: int) -> FloatArray:
    """Add shared-channel, midtone-weighted noise using a local seeded generator."""
    if amount == 0:
        return rgb.copy()
    luma = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    weight = np.sqrt(np.maximum(4 * luma * (1 - luma), 0))
    noise = np.random.default_rng(seed).normal(0, amount, luma.shape).astype(np.float32)
    return np.clip(rgb + (noise * weight)[..., None], 0, 1)
