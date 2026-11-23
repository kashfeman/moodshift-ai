"""Offline checks for the trainable decoder and its frozen encoder."""

import math

import pytest
import torch
from torchvision.models import squeezenet1_1

from moodshift.engine.features import SqueezeNetFeatures
from moodshift.engine.training import (
    EncoderAdapter,
    ImageDecoder,
    TrainingConfig,
    align_features,
    make_optimizer,
    training_step,
)


def test_training_step_updates_only_decoder() -> None:
    """Backpropagate through the real architecture without downloading weights."""
    torch.manual_seed(13)
    encoder = EncoderAdapter(SqueezeNetFeatures(squeezenet1_1(weights=None).features))
    decoder = ImageDecoder()
    content = torch.rand(1, 3, 33, 41)
    style = torch.rand(1, 3, 40, 35)
    features = encoder(content)
    assert features.shape[1:] == (384, 2, 2)
    assert decoder(features, (33, 41)).shape == content.shape
    encoder_before = [p.detach().clone() for p in encoder.parameters()]
    decoder_before = [p.detach().clone() for p in decoder.parameters()]
    config = TrainingConfig()
    losses = training_step(
        encoder, decoder, make_optimizer(decoder, config), content, style, config
    )
    assert all(math.isfinite(value) for value in losses.values())
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in decoder.parameters()
    )
    assert any(
        not torch.equal(before, after)
        for before, after in zip(decoder_before, decoder.parameters(), strict=True)
    )
    assert all(
        torch.equal(before, after) and after.grad is None
        for before, after in zip(encoder_before, encoder.parameters(), strict=True)
    )


def test_alignment_and_invalid_inputs() -> None:
    """Support differing spatial sizes and reject invalid or nonfinite tensors."""
    content = torch.randn(2, 384, 3, 4, requires_grad=True)
    style = torch.randn(2, 384, 5, 6)
    aligned = align_features(content, style)
    assert aligned.shape == content.shape and torch.isfinite(aligned).all()
    aligned.square().mean().backward()
    assert content.grad is not None and torch.isfinite(content.grad).all()
    assert torch.isfinite(align_features(torch.ones_like(content), style)).all()
    with pytest.raises(ValueError):
        align_features(content, style[:1])
    with pytest.raises(ValueError):
        ImageDecoder()(torch.full((1, 384, 2, 2), float("nan")), (17, 17))
    with pytest.raises(ValueError):
        TrainingConfig(learning_rate=0)
    with pytest.raises(ValueError):
        EncoderAdapter(SqueezeNetFeatures(squeezenet1_1(weights=None).features))(
            torch.rand(1, 3, 16, 20)
        )
