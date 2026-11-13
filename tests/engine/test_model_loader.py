"""Loader failure and frozen feature tests never download pretrained weights."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from moodshift.engine import model_loader
from moodshift.engine.exceptions import DeviceError, ModelLoadError
from moodshift.engine.features import SqueezeNetFeatures


def test_device_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Auto uses actual CUDA availability while explicit unavailable CUDA fails."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert model_loader.select_device().type == "cpu"
    with pytest.raises(DeviceError):
        model_loader.select_device("cuda")
    with pytest.raises(ValueError):
        model_loader.select_device("invalid")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert model_loader.select_device().type == "cuda"
    assert model_loader.select_device("cpu").type == "cpu"


def test_missing_or_corrupted_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Offline missing and corrupt cached checkpoints fail without replacement."""
    checkpoint = tmp_path / "squeezenet1_1-b8a52dc0.pth"
    monkeypatch.setattr(model_loader, "pretrained_checkpoint_path", lambda: checkpoint)
    with pytest.raises(ModelLoadError, match="not cached"):
        model_loader.load_feature_extractor(torch.device("cpu"), allow_download=False)
    checkpoint.write_bytes(b"not weights")
    with pytest.raises(ModelLoadError, match="checksum"):
        model_loader.load_feature_extractor(torch.device("cpu"))
    assert checkpoint.read_bytes() == b"not weights"


def test_download_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Network failure is surfaced without returning an untrained model."""
    monkeypatch.setattr(
        model_loader, "pretrained_checkpoint_path", lambda: tmp_path / "missing"
    )

    def failing_builder(**kwargs: object) -> None:
        raise OSError("offline")

    monkeypatch.setattr(model_loader, "squeezenet1_1", failing_builder)
    with pytest.raises(ModelLoadError, match="Unable to load"):
        model_loader.load_feature_extractor(torch.device("cpu"))


def test_feature_maps_frozen_and_differentiable() -> None:
    """Frozen intermediate extraction retains finite gradients to input pixels."""
    with torch.random.fork_rng(devices=[]):
        layers = nn.Sequential(nn.Conv2d(3, 3, 1), *[nn.ReLU() for _ in range(9)])
    extractor = SqueezeNetFeatures(layers)
    assert not extractor.training
    assert all(not parameter.requires_grad for parameter in extractor.parameters())
    pixels = torch.ones(1, 3, 20, 20, requires_grad=True)
    features = extractor(pixels)
    assert set(features) == {"1", "3", "6", "9"}
    sum(value.sum() for value in features.values()).backward()
    assert pixels.grad is not None and torch.isfinite(pixels.grad).all()


def test_builder_uses_default_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Production builder always requests official weights and frozen eval features."""
    monkeypatch.setattr(
        model_loader, "pretrained_checkpoint_path", lambda: tmp_path / "missing"
    )
    calls = []

    def builder(**kwargs: object) -> SimpleNamespace:
        calls.append(kwargs)
        return SimpleNamespace(
            features=nn.Sequential(*[nn.Identity() for _ in range(13)])
        )

    monkeypatch.setattr(model_loader, "squeezenet1_1", builder)
    extractor = model_loader.load_feature_extractor(torch.device("cpu"))
    assert calls[0]["weights"] is model_loader.SqueezeNet1_1_Weights.DEFAULT
    assert not extractor.training
    assert len(extractor.layers) == 10
