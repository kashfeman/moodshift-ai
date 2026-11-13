"""Bounded controls and pixel normalization/reconstruction contracts."""

import numpy as np
import pytest
import torch
from PIL import Image

from moodshift.engine.settings import TransferSettings
from moodshift.engine.tensors import image_to_tensor, normalize_tensor, tensor_to_image


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("style_strength", -0.1),
        ("style_strength", 1.1),
        ("style_strength", float("nan")),
        ("style_strength", True),
        ("content_preservation", float("inf")),
        ("content_preservation", "0.5"),
        ("content_preservation", -1),
        ("steps", 0),
        ("steps", 501),
        ("steps", True),
        ("steps", 2.5),
        ("max_size", 31),
        ("max_size", 513),
        ("seed", -1),
        ("seed", 2**32),
        ("seed", True),
    ],
)
def test_invalid_settings(name: str, value: object) -> None:
    """Settings reject nonfinite, invalid types, and unsafe bounds."""
    with pytest.raises(ValueError, match=name):
        TransferSettings(**{name: value})


@pytest.mark.parametrize("value", [0.0, 0.3, 1.0])
def test_valid_strengths(value: float) -> None:
    """Both controls accept inclusive endpoints and intermediate values."""
    assert TransferSettings(style_strength=value, content_preservation=value)


def test_quality_modes() -> None:
    """High quality uses more steps and pixels than the laptop preview default."""
    preview = TransferSettings.for_quality("preview")
    high = TransferSettings.for_quality("high")
    assert preview.steps < high.steps <= 500
    assert preview.max_size < high.max_size <= 512
    with pytest.raises(ValueError):
        TransferSettings.for_quality("unknown")


def test_tensor_normalization_and_reconstruction() -> None:
    """RGB channel order and ImageNet scale are exact without input mutation."""
    array = np.array([[[255, 128, 0], [0, 255, 255]]], dtype=np.uint8)
    image = Image.fromarray(array)
    pixels = image_to_tensor(image, torch.device("cpu"))
    assert pixels.shape == (1, 3, 1, 2)
    assert pixels.dtype == torch.float32
    assert pixels[0, :, 0, 0].tolist() == pytest.approx([1.0, 128 / 255, 0.0])
    expected = (
        pixels - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    ) / torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    assert torch.allclose(normalize_tensor(pixels), expected)
    assert np.array_equal(np.asarray(tensor_to_image(pixels)), array)
    assert np.array_equal(np.asarray(image), array)
    pixels.requires_grad_(True)
    normalize_tensor(pixels).sum().backward()
    assert torch.isfinite(pixels.grad).all()


@pytest.mark.parametrize(
    "pixels",
    [
        torch.full((1, 3, 2, 2), float("nan")),
        torch.zeros(3, 2, 2),
        torch.zeros(1, 4, 2, 2),
    ],
)
def test_invalid_output_tensor(pixels: torch.Tensor) -> None:
    """Malformed or nonfinite output cannot become a plausible RGB image."""
    with pytest.raises(ValueError):
        tensor_to_image(pixels)
