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


def test_objective_reconstruction_and_validation_agree() -> None:
    """Train and validation share finite normalized losses and RGB reconstruction."""
    from moodshift.engine.train_decoder import evaluate_batch
    from moodshift.engine.training import decoder_objective

    torch.manual_seed(42)
    encoder = EncoderAdapter(SqueezeNetFeatures(squeezenet1_1(weights=None).features))
    decoder = ImageDecoder()
    content = torch.rand(1, 3, 32, 32)
    style = torch.rand(1, 3, 32, 32)
    config = TrainingConfig()
    losses = decoder_objective(encoder, decoder, content, style, config)
    measured = evaluate_batch(encoder, decoder, content, style, config)
    assert losses["reconstruction_loss"].item() > 0
    assert all(math.isfinite(value.item()) for value in losses.values())
    assert all(
        math.isclose(value.item(), measured[key]) for key, value in losses.items()
    )
    losses["loss"].backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in decoder.parameters()
    )
    assert all(p.grad is None for p in encoder.parameters())


def test_official_normalization_and_rgb_order() -> None:
    """Match TorchVision's documented SqueezeNet normalization for RGB pixels."""
    from torchvision.models import SqueezeNet1_1_Weights

    from moodshift.engine.tensors import normalize_tensor

    preprocess = SqueezeNet1_1_Weights.DEFAULT.transforms()
    pixels = torch.tensor([0.2, 0.5, 0.8]).view(1, 3, 1, 1)
    mean = torch.tensor(preprocess.mean).view(1, 3, 1, 1)
    std = torch.tensor(preprocess.std).view(1, 3, 1, 1)
    torch.testing.assert_close(normalize_tensor(pixels), (pixels - mean) / std)


def test_training_clips_large_gradients() -> None:
    """The actual optimizer step limits the aggregate decoder gradient norm."""
    torch.manual_seed(7)
    encoder = EncoderAdapter(SqueezeNetFeatures(squeezenet1_1(weights=None).features))
    decoder = ImageDecoder()
    config = TrainingConfig(reconstruction_weight=1000)
    pixels = torch.rand(1, 3, 32, 32)
    training_step(
        encoder, decoder, make_optimizer(decoder, config), pixels, pixels, config
    )
    norm = torch.linalg.vector_norm(
        torch.stack([p.grad.norm() for p in decoder.parameters() if p.grad is not None])
    )
    assert norm <= 1.00001
