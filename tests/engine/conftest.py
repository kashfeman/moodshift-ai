"""Small differentiable fixtures keep engine unit tests offline and CPU-only."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import Tensor, nn
from torch.nn import functional as F


class TinyFeatures(nn.Module):
    """Fixed convolutional feature test double; never used in production."""

    def __init__(self) -> None:
        """Initialize a deterministic, frozen convolution without changing RNG."""
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(5)
            self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.requires_grad_(False)
        self.eval()

    def forward(self, pixels: Tensor) -> dict[str, Tensor]:
        """Return maps at several scales with differentiable channel mixing."""
        value = self.conv(pixels).tanh()
        return {
            "1": value,
            "3": value.square(),
            "6": F.avg_pool2d(value, 2),
            "9": F.avg_pool2d(value, 4),
        }


@pytest.fixture(scope="session", autouse=True)
def cpu_threads() -> Iterator[None]:
    """Avoid CPU oversubscription in tiny tests and restore caller settings."""
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def tiny_features() -> TinyFeatures:
    """Provide a frozen differentiable test extractor."""
    return TinyFeatures()


@pytest.fixture
def image_paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create a structured content image and two visibly different references."""
    y, x = np.indices((40, 48))
    content = np.stack((x * 5, y * 6, ((x + y) % 8) * 32), axis=-1).astype(np.uint8)
    style = np.random.default_rng(17).integers(0, 256, (32, 44, 3), dtype=np.uint8)
    other = np.zeros((36, 40, 3), dtype=np.uint8)
    other[..., 2] = 220
    paths = tuple(tmp_path / name for name in ("content.png", "style.png", "other.png"))
    for path, array in zip(paths, (content, style, other), strict=True):
        Image.fromarray(array).save(path)
    return paths
