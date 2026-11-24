"""Transfer a trained decoder's low-frequency changes onto its source crop."""

import math

import cv2
import numpy as np
from PIL import Image, ImageOps

from moodshift.imaging.color import lab_to_rgb, pil_to_rgb_array, rgb_to_lab


def center_crop_for_decoder(image: Image.Image) -> Image.Image:
    """Return the full-resolution square region used by centered ImageOps.fit.

    Half-pixel crop offsets are sampled with bicubic interpolation, matching the
    center of Pillow's fractional crop rather than shifting by a whole pixel.
    """
    side = min(image.size)
    left = (image.width - side) / 2
    top = (image.height - side) / 2
    if left.is_integer() and top.is_integer():
        return image.crop((int(left), int(top), int(left) + side, int(top) + side))
    return image.transform(
        (side, side),
        Image.Transform.EXTENT,
        (left, top, left + side, top + side),
        resample=Image.Resampling.BICUBIC,
    )


def reconstruct_decoder_color(
    source_crop: Image.Image,
    decoder_input: Image.Image,
    decoder_output: Image.Image,
    amplification: float = 1.0,
) -> Image.Image:
    """Apply smoothed bounded LAB differences to the matching source crop.

    The decoder inputs and output must have identical dimensions and represent
    the same centered crop. This transfers only its changes, without inventing a
    unrelated effects. Amplification is limited to 1–2; a zero residual returns
    the original pixels exactly.
    """
    if (
        isinstance(amplification, bool)
        or not isinstance(amplification, (int, float))
        or not math.isfinite(amplification)
        or not 1.0 <= amplification <= 2.0
    ):
        raise ValueError("Reconstruction amplification must be between 1 and 2")
    original = pil_to_rgb_array(source_crop)
    before = pil_to_rgb_array(decoder_input)
    after = pil_to_rgb_array(decoder_output)
    if before.shape != after.shape or decoder_input.width != decoder_input.height:
        raise ValueError(
            "Decoder input and output must have matching square dimensions"
        )
    if source_crop.width != source_crop.height:
        raise ValueError(
            "Source crop must be square and aligned with the decoder input"
        )
    if np.array_equal(before, after):
        return source_crop.copy()
    delta = rgb_to_lab(after) - rgb_to_lab(before)
    delta = cv2.GaussianBlur(delta, (0, 0), sigmaX=2, borderType=cv2.BORDER_REFLECT)
    delta = np.clip(delta * amplification, [-12, -15, -15], [12, 15, 15]).astype(
        np.float32
    )
    delta = cv2.resize(delta, source_crop.size, interpolation=cv2.INTER_LINEAR)
    result = rgb_to_lab(original) + delta
    result[..., 0] = np.clip(result[..., 0], 0, 100)
    return Image.fromarray(lab_to_rgb(result.astype(np.float32)))


def decoder_input_from_original(image: Image.Image) -> Image.Image:
    """Keep the existing decoder preprocessing byte-for-byte unchanged."""
    return ImageOps.fit(image, (96, 96), method=Image.Resampling.LANCZOS)
