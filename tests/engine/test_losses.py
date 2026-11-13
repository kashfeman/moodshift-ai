"""Exact small-tensor loss checks independent of models and downloads."""

import pytest
import torch

from moodshift.engine.losses import (
    content_loss,
    gram_matrix,
    style_loss,
    total_variation_loss,
)


def test_gram_normalization_and_batch_separation() -> None:
    """Channel correlations divide by C*H*W without mixing separate images."""
    features = torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]]]])
    expected = torch.tensor([[[5.0, 11.0], [11.0, 25.0]]]) / 4
    assert torch.allclose(gram_matrix(features), expected)
    batch = torch.cat((features, features * 2))
    assert torch.allclose(gram_matrix(batch)[0], expected[0])
    assert torch.allclose(gram_matrix(batch)[1], expected[0] * 4)
    assert torch.allclose(gram_matrix(features.flip(-1)), expected)


def test_content_loss_and_shape_rejection() -> None:
    """Content is spatial MSE with exact zero for identical features."""
    features = torch.ones(1, 2, 2, 2)
    assert content_loss(features, features) == 0
    assert content_loss(features, features * 3) == 4
    with pytest.raises(ValueError):
        content_loss(features, torch.ones(1, 2, 1, 1))


def test_style_loss() -> None:
    """Style compares normalized Gram matrices and accepts different image sizes."""
    features = torch.ones(1, 2, 2, 2)
    target = gram_matrix(features)
    assert style_loss(features, target) == 0
    assert style_loss(torch.ones(1, 2, 3, 5), target) == 0
    assert style_loss(features * 2, target) == pytest.approx(2.25)
    with pytest.raises(ValueError):
        style_loss(features, torch.ones(1, 3, 3))


def test_tv_and_finite_gradients() -> None:
    """TV uses both directions and differentiates finitely on singleton axes."""
    pixels = torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]], requires_grad=True)
    assert total_variation_loss(pixels).item() == 5
    loss = (
        total_variation_loss(pixels)
        + content_loss(pixels, torch.zeros_like(pixels))
        + style_loss(pixels, gram_matrix(torch.ones_like(pixels)))
    )
    loss.backward()
    assert pixels.grad is not None and torch.isfinite(pixels.grad).all()
    assert total_variation_loss(torch.ones(1, 3, 1, 1)) == 0
    assert total_variation_loss(torch.ones(1, 3, 3, 3)) == 0


@pytest.mark.parametrize("features", [torch.ones(2, 2), torch.empty(1, 3, 0, 2)])
def test_invalid_loss_shapes(features: torch.Tensor) -> None:
    """Non-image and empty shapes fail explicitly."""
    with pytest.raises(ValueError):
        gram_matrix(features)
    with pytest.raises(ValueError):
        total_variation_loss(features)
