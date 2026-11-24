"""Content-guided reconstruction of low-resolution neural color changes.

Hybrid mode transfers an edge-aware color residual instead of enlarging generated
texture. Original pixels supply detail; no super-resolution or sharpening occurs.
The local linear guided-filter equations follow He, Sun and Tang (ECCV 2010).
"""

import cv2
import numpy as np
from PIL import Image

from moodshift.imaging.color import pil_to_rgb_array, rgb_array_to_pil


def reconstruct_hybrid(content: Image.Image, neural: Image.Image) -> Image.Image:
    """Apply a guided low-resolution neural RGB residual to original RGB detail.

    Fit residual = a * luminance + b in overlapping 9-pixel windows, with
    regularization 0.01 on normalized luminance. Upsample averaged coefficients
    rather than neural RGB pixels. This favors scene structure over style texture;
    clipping can still change detail at gamut boundaries. Inputs remain unchanged.
    """
    original = pil_to_rgb_array(content).astype(np.float32) / 255
    generated = pil_to_rgb_array(neural).astype(np.float32) / 255
    reduced = content.resize(neural.size, Image.Resampling.LANCZOS)
    base = pil_to_rgb_array(reduced).astype(np.float32) / 255
    guide = cv2.cvtColor(base, cv2.COLOR_RGB2GRAY)
    full_guide = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    residual = generated - base

    def mean(array: np.ndarray) -> np.ndarray:
        return cv2.boxFilter(array, -1, (9, 9), borderType=cv2.BORDER_REFLECT)

    guide_mean = mean(guide)
    variance = np.maximum(mean(guide * guide) - guide_mean**2, 0)
    correction = np.empty_like(original)
    for channel in range(3):
        signal = residual[..., channel]
        signal_mean = mean(signal)
        covariance = mean(guide * signal) - guide_mean * signal_mean
        slope = covariance / (variance + 0.01)
        intercept = signal_mean - slope * guide_mean
        slope = cv2.resize(mean(slope), content.size, interpolation=cv2.INTER_LINEAR)
        intercept = cv2.resize(
            mean(intercept), content.size, interpolation=cv2.INTER_LINEAR
        )
        correction[..., channel] = slope * full_guide + intercept
    pixels = np.rint(np.clip(original + correction, 0, 1) * 255).astype(np.uint8)
    return rgb_array_to_pil(pixels)


def reconstruct_neural(content: Image.Image, neural: Image.Image) -> Image.Image:
    """Combine neural texture with a stronger source-detail-guided reconstruction.

    The direct neural image contributes 30 percent, retaining some optimized
    texture; the guided residual supplies 70 percent. No neural inference is
    substituted, and both paths use the same genuine neural output.
    """
    guided = reconstruct_hybrid(content, neural)
    direct = neural.resize(content.size, Image.Resampling.LANCZOS)
    return Image.blend(guided, direct, 0.30)
