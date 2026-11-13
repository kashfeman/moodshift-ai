"""RGB pixel tensors and ImageNet normalization without crops or distortion."""

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from moodshift.imaging.color import pil_to_rgb_array, rgb_array_to_pil


def image_to_tensor(image: Image.Image, device: torch.device) -> Tensor:
    """Copy RGB pixels to float32 1 x 3 x H x W in [0, 1]."""
    array = pil_to_rgb_array(image)
    return (
        torch.from_numpy(array)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
        / 255.0
    )


def normalize_tensor(pixels: Tensor) -> Tensor:
    """Apply pretrained ImageNet RGB mean/std while preserving pixel gradients."""
    mean = pixels.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = pixels.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    return (pixels - mean) / std


def tensor_to_image(pixels: Tensor) -> Image.Image:
    """Detach finite float RGB pixels, clamp, round, and copy to uint8 RGB."""
    if (
        pixels.ndim != 4
        or pixels.shape[0] != 1
        or pixels.shape[1] != 3
        or min(pixels.shape[2:]) < 1
        or not pixels.is_floating_point()
        or not torch.isfinite(pixels).all()
    ):
        raise ValueError("Expected finite floating 1 x 3 x H x W RGB pixels")
    array = (
        pixels.detach()
        .cpu()
        .clamp(0, 1)
        .squeeze(0)
        .permute(1, 2, 0)
        .mul(255)
        .round()
        .to(torch.uint8)
        .numpy()
    )
    return rgb_array_to_pil(np.array(array, copy=True))
