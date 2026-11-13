"""Intermediate SqueezeNet feature maps; classifier outputs are never used."""

from collections.abc import Mapping
from typing import Protocol

from torch import Tensor, nn

from moodshift.engine.tensors import normalize_tensor

STYLE_LAYERS = ("1", "3", "6", "9")
CONTENT_LAYER = "9"


class FeatureExtractor(Protocol):
    """Callable feature boundary used by the optimizer and isolated tests."""

    def __call__(self, pixels: Tensor) -> Mapping[str, Tensor]:
        """Extract the configured feature maps from RGB pixels in [0, 1]."""
        ...


class SqueezeNetFeatures(nn.Module):
    """Capture conv/ReLU and Fire outputs through Fire6 (feature index 9)."""

    def __init__(self, layers: nn.Sequential) -> None:
        """Own the selected pretrained feature prefix, frozen in evaluation mode."""
        super().__init__()
        self.layers = layers[:10]
        self.requires_grad_(False)
        self.eval()

    def forward(self, pixels: Tensor) -> dict[str, Tensor]:
        """Extract ImageNet-normalized maps while retaining input gradients."""
        value = normalize_tensor(pixels)
        result = {}
        for index, layer in enumerate(self.layers):
            value = layer(value)
            if str(index) in STYLE_LAYERS:
                result[str(index)] = value
        return result
