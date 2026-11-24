"""Tests for the separate RGB reference conditioning branch."""

import pytest
import torch
from torchvision.models import squeezenet1_1

from moodshift.engine.color_conditioned_decoder import (
    ColorConditionedDecoder,
    color_matching_objective,
    color_matching_step,
    color_objective,
    color_step,
    rgb_statistics_loss,
)
from moodshift.engine.features import SqueezeNetFeatures
from moodshift.engine.style_conditioned_decoder import StyleConditionedDecoder
from moodshift.engine.train_guided import model_fingerprint
from moodshift.engine.training import EncoderAdapter, align_features


def test_rgb_conditioning_is_connected_and_trainable() -> None:
    """Vary RGB reference only and verify gradients and frozen encoder weights."""
    torch.manual_seed(12)
    encoder = EncoderAdapter(SqueezeNetFeatures(squeezenet1_1(weights=None).features))
    decoder = ColorConditionedDecoder()
    content = torch.rand(1, 3, 32, 40)
    style_a = torch.rand(1, 3, 32, 40) * 0.4
    style_b = torch.rand(1, 3, 32, 40) * 0.4 + 0.6
    with torch.no_grad():
        sf = encoder(style_a)
        target = align_features(encoder(content), sf)
        a = decoder(target, content, sf, style_a)
        b = decoder(target, content, sf, style_b)
    assert a.shape == content.shape and torch.isfinite(a).all()
    assert 0 <= a.min() <= a.max() <= 1
    assert (a - b).abs().max() > 1e-6
    before = [p.detach().clone() for p in decoder.parameters()]
    enc_before = [p.detach().clone() for p in encoder.parameters()]
    metrics = color_step(
        encoder,
        decoder,
        torch.optim.Adam(decoder.parameters(), lr=1e-4),
        content,
        style_a,
    )
    assert all(torch.isfinite(torch.tensor(x)) for x in metrics.values())
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in decoder.rgb_condition.parameters()
    )
    assert any(
        not torch.equal(a, b) for a, b in zip(before, decoder.parameters(), strict=True)
    )
    assert all(
        torch.equal(a, b) and b.grad is None
        for a, b in zip(enc_before, encoder.parameters(), strict=True)
    )


def test_same_initial_base_and_input_validation() -> None:
    """Verify the original conditioned decoder and its initialization remain intact."""
    torch.manual_seed(0)
    old = StyleConditionedDecoder()
    torch.manual_seed(0)
    new = ColorConditionedDecoder()
    assert model_fingerprint(old) == model_fingerprint(new.base)
    with pytest.raises(ValueError):
        new(
            torch.rand(1, 384, 2, 2),
            torch.rand(1, 3, 32, 32),
            torch.rand(1, 384, 2, 2),
            torch.full((1, 3, 32, 32), 2.0),
        )


def test_rgb_statistics_loss_gradients_and_identity() -> None:
    """The color loss is smaller for a matching reference and differentiable."""
    torch.manual_seed(42)
    reference = torch.rand(1, 3, 24, 32)
    output = torch.full((1, 3, 24, 32), 0.2, requires_grad=True)
    loss = rgb_statistics_loss(output, reference)
    assert rgb_statistics_loss(reference, reference) < loss
    loss.backward()
    assert output.grad is not None and torch.isfinite(output.grad).all()
    assert output.grad.abs().sum() > 0


def test_rgb_loss_training_updates_decoder_but_not_encoder() -> None:
    """The new objective trains only decoder parameters through frozen features."""
    torch.manual_seed(17)
    encoder = EncoderAdapter(SqueezeNetFeatures(squeezenet1_1(weights=None).features))
    decoder = ColorConditionedDecoder()
    content, style = torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32)
    original = [p.detach().clone() for p in encoder.parameters()]
    initial = [p.detach().clone() for p in decoder.parameters()]
    objective = color_matching_objective(encoder, decoder, content, style)
    assert torch.isfinite(objective["rgb_color_loss"])
    baseline = color_objective(encoder, decoder, content, style)
    torch.testing.assert_close(
        objective["loss"], baseline["loss"] + objective["rgb_color_loss"]
    )
    result = color_matching_step(
        encoder,
        decoder,
        torch.optim.Adam(decoder.parameters(), lr=1e-4),
        content,
        style,
    )
    assert all(torch.isfinite(torch.tensor(value)) for value in result.values())
    assert any(
        not torch.equal(a, b)
        for a, b in zip(initial, decoder.parameters(), strict=True)
    )
    assert all(
        torch.equal(a, b) and b.grad is None
        for a, b in zip(original, encoder.parameters(), strict=True)
    )
