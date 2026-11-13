"""Isolated optimizer tests use only deterministic differentiable test features."""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from moodshift.engine import neural_transfer
from moodshift.engine.demo import main
from moodshift.engine.exceptions import NeuralTransferError
from moodshift.engine.features import FeatureExtractor
from moodshift.engine.neural_transfer import NeuralStyleTransfer
from moodshift.engine.settings import TransferSettings
from moodshift.imaging.exceptions import InvalidImageError


def test_zero_strength_no_resize_or_model(
    image_paths: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Zero strength preserves full decoded RGB exactly without loading weights."""

    def unexpected_load(*args: object, **kwargs: object) -> None:
        raise AssertionError("Zero strength must not load a model")

    monkeypatch.setattr(neural_transfer, "load_feature_extractor", unexpected_load)
    content, style, _ = image_paths
    original = content.read_bytes()
    result = NeuralStyleTransfer(device="cpu").transfer(
        content, style, TransferSettings(style_strength=0, max_size=32)
    )
    with Image.open(content) as image:
        assert np.array_equal(np.asarray(result), np.asarray(image))
        assert result.size == image.size
    assert content.read_bytes() == original


@pytest.mark.parametrize(
    ("strength", "preservation"), [(0.2, 1.0), (1.0, 0.0), (0.6, 0.5)]
)
def test_cpu_output_and_reference_influence(
    image_paths: tuple[Path, Path, Path],
    tiny_features: FeatureExtractor,
    strength: float,
    preservation: float,
) -> None:
    """Optimization responds to references without modifying input files."""
    content, style, other = image_paths
    before = [path.read_bytes() for path in image_paths]
    engine = NeuralStyleTransfer(device="cpu", extractor=tiny_features)
    settings = TransferSettings(
        style_strength=strength, content_preservation=preservation, steps=5, max_size=48
    )
    state = torch.random.get_rng_state().clone()
    first = engine.transfer(content, style, settings)
    second = engine.transfer(content, other, settings)
    repeated = engine.transfer(content, style, settings)
    assert engine.device.type == "cpu"
    assert first.mode == "RGB" and first.size == (48, 40)
    array = np.asarray(first)
    assert np.isfinite(array).all() and array.min() >= 0 and array.max() <= 255
    assert np.array_equal(array, np.asarray(repeated))
    assert not np.array_equal(array, np.asarray(second))
    with Image.open(content) as image:
        assert not np.array_equal(array, np.asarray(image))
    assert torch.equal(state, torch.random.get_rng_state())
    assert [path.read_bytes() for path in image_paths] == before


def test_invalid_inputs_and_tiny_images(
    image_paths: tuple[Path, Path, Path],
    tiny_features: FeatureExtractor,
    tmp_path: Path,
) -> None:
    """Existing image validation and CNN minimum dimensions fail explicitly."""
    content, style, _ = image_paths
    engine = NeuralStyleTransfer(device="cpu", extractor=tiny_features)
    with pytest.raises(InvalidImageError):
        engine.transfer(tmp_path / "missing.png", style)
    with pytest.raises(InvalidImageError):
        engine.transfer(content, tmp_path / "missing.png")
    tiny = tmp_path / "tiny.png"
    Image.new("RGB", (8, 30)).save(tiny)
    with pytest.raises(ValueError, match="17px"):
        engine.transfer(content, tiny)


def test_nonfinite_and_memory_failure(image_paths: tuple[Path, Path, Path]) -> None:
    """Failures return typed errors instead of an unchanged or fabricated image."""

    def nonfinite(pixels: torch.Tensor) -> dict[str, torch.Tensor]:
        return {name: pixels * float("nan") for name in ("1", "3", "6", "9")}

    def memory_failure(pixels: torch.Tensor) -> dict[str, torch.Tensor]:
        raise MemoryError("test allocation failure")

    content, style, _ = image_paths
    for extractor, message in ((nonfinite, "Nonfinite"), (memory_failure, "failed")):
        with pytest.raises(NeuralTransferError, match=message):
            NeuralStyleTransfer(device="cpu", extractor=extractor).transfer(
                content, style
            )


def test_lazy_model_reuse(
    image_paths: tuple[Path, Path, Path],
    tiny_features: FeatureExtractor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reusable engine loads its model once, not once per generated image."""
    calls = []

    def loader(*args: object, **kwargs: object) -> FeatureExtractor:
        calls.append(True)
        return tiny_features

    monkeypatch.setattr(neural_transfer, "load_feature_extractor", loader)
    engine = NeuralStyleTransfer(device="cpu")
    assert not calls
    content, style, _ = image_paths
    for _ in range(2):
        engine.transfer(content, style, TransferSettings(steps=1))
    assert len(calls) == 1


def test_nonfinite_gradient_rejected(image_paths: tuple[Path, Path, Path]) -> None:
    """Finite forward values cannot hide a nonfinite backward gradient."""

    def invalid_backward(pixels: torch.Tensor) -> dict[str, torch.Tensor]:
        if pixels.requires_grad:
            pixels.register_hook(lambda grad: torch.full_like(grad, float("nan")))
        return {name: pixels for name in ("1", "3", "6", "9")}

    content, style, _ = image_paths
    engine = NeuralStyleTransfer(device="cpu", extractor=invalid_backward)
    with pytest.raises(NeuralTransferError, match="Nonfinite neural gradient"):
        engine.transfer(content, style, TransferSettings(steps=1))


def test_nonfinite_optimizer_output_rejected(
    image_paths: tuple[Path, Path, Path],
    tiny_features: FeatureExtractor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid optimizer output is rejected before RGB clamping could hide it."""

    class BrokenAdam(torch.optim.Adam):
        def step(self, closure: object = None) -> None:
            with torch.no_grad():
                self.param_groups[0]["params"][0].fill_(float("inf"))

    monkeypatch.setattr(torch.optim, "Adam", BrokenAdam)
    content, style, _ = image_paths
    engine = NeuralStyleTransfer(device="cpu", extractor=tiny_features)
    with pytest.raises(NeuralTransferError, match="Nonfinite neural output"):
        engine.transfer(content, style, TransferSettings(steps=1))


def test_cli_zero_strength(
    image_paths: tuple[Path, Path, Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI uses safe export and reports the actual device without downloads."""
    content, style, _ = image_paths
    args = [
        "--content",
        str(content),
        "--style",
        str(style),
        "--output",
        str(tmp_path / "result.png"),
        "--strength",
        "0",
        "--device",
        "cpu",
    ]
    assert main(args) == 0
    assert "Completed on cpu" in capsys.readouterr().out
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 1
