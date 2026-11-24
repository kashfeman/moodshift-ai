"""Tests for the experimental guided decoder using offline model fixtures."""

import math
from pathlib import Path

import pytest
import torch
from torchvision.models import squeezenet1_1

from moodshift.engine.features import SqueezeNetFeatures
from moodshift.engine.guided_decoder import (
    BASELINE_WEIGHTS,
    STRONGER_STYLE_WEIGHTS,
    GuidedDecoder,
    GuidedLossWeights,
    guided_objective,
    guided_step,
)
from moodshift.engine.training import EncoderAdapter, align_features


def test_guided_updates_frozen_encoder_and_style_influence(tmp_path: Path) -> None:
    """Train, reload and verify RGB shape, finite gradients, and style dependence."""
    torch.manual_seed(12)
    encoder = EncoderAdapter(SqueezeNetFeatures(squeezenet1_1(weights=None).features))
    model = GuidedDecoder()
    content = torch.rand(1, 3, 32, 40)
    style = torch.rand(1, 3, 32, 40)
    other = torch.rand(1, 3, 32, 40)
    decoder_before = [p.detach().clone() for p in model.parameters()]
    encoder_before = [p.detach().clone() for p in encoder.parameters()]
    losses = guided_step(
        encoder,
        model,
        torch.optim.Adam(model.parameters(), lr=1e-4),
        content,
        style,
        STRONGER_STYLE_WEIGHTS,
    )
    assert all(math.isfinite(value) for value in losses.values())
    assert any(
        not torch.equal(a, b)
        for a, b in zip(decoder_before, model.parameters(), strict=True)
    )
    assert all(
        torch.equal(a, b) and b.grad is None
        for a, b in zip(encoder_before, encoder.parameters(), strict=True)
    )
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
    )
    with torch.no_grad():
        content_features = encoder(content)
        aligned = align_features(content_features, encoder(style))
        alt = align_features(content_features, encoder(other))
        result = model(aligned, content)
        assert result.shape == content.shape
        assert torch.isfinite(result).all() and 0 <= result.min() <= result.max() <= 1
        assert (result - model(alt, content)).abs().max() > 1e-7
        path = tmp_path / "checkpoint.pt"
        torch.save({"decoder": model.state_dict()}, path)
        reloaded = GuidedDecoder()
        reloaded.load_state_dict(torch.load(path, weights_only=True)["decoder"])
        torch.testing.assert_close(result, reloaded(aligned, content))


def test_invalid_guided_inputs() -> None:
    """Reject incompatible channels and RGB range."""
    model = GuidedDecoder()
    with pytest.raises(ValueError):
        model(torch.rand(1, 20, 3, 3), torch.rand(1, 3, 32, 32))
    with pytest.raises(ValueError):
        model(torch.rand(1, 384, 3, 3), torch.full((1, 3, 32, 32), 2.0))


def test_guided_loss_presets_use_correct_terms() -> None:
    """The presets only change the weighting, not the underlying loss terms."""
    torch.manual_seed(10)
    encoder = EncoderAdapter(SqueezeNetFeatures(squeezenet1_1(weights=None).features))
    decoder = GuidedDecoder()
    content = torch.rand(1, 3, 32, 32)
    style = torch.rand(1, 3, 32, 32)
    baseline = guided_objective(encoder, decoder, content, style)
    experimental = guided_objective(
        encoder, decoder, content, style, STRONGER_STYLE_WEIGHTS
    )
    assert BASELINE_WEIGHTS == GuidedLossWeights(1, 1, 10, 2)
    assert STRONGER_STYLE_WEIGHTS == GuidedLossWeights(1, 3, 5, 1)
    for name in ("content_loss", "style_loss", "reconstruction_loss", "edge_loss"):
        torch.testing.assert_close(baseline[name], experimental[name])
        assert torch.isfinite(baseline[name])
    torch.testing.assert_close(
        baseline["loss"],
        baseline["content_loss"]
        + baseline["style_loss"]
        + 10 * baseline["reconstruction_loss"]
        + 2 * baseline["edge_loss"],
    )
    torch.testing.assert_close(
        experimental["loss"],
        experimental["content_loss"]
        + 3 * experimental["style_loss"]
        + 5 * experimental["reconstruction_loss"]
        + experimental["edge_loss"],
    )


@pytest.mark.parametrize("bad", [-1, float("nan"), float("inf"), True, "1"])
def test_invalid_loss_weights(bad: object) -> None:
    """Reject negative, nonfinite, boolean, and nonnumeric weights."""
    with pytest.raises(ValueError, match="loss weights"):
        GuidedLossWeights(style=bad)  # type: ignore[arg-type]


def test_all_zero_loss_weights_are_invalid() -> None:
    """At least one objective term must be active."""
    with pytest.raises(ValueError, match="at least one"):
        GuidedLossWeights(0, 0, 0, 0)
