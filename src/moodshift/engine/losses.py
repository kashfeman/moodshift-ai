"""Differentiable neural content, Gram style, and total variation losses."""

import torch
from torch import Tensor
from torch.nn import functional as F


def gram_matrix(features: Tensor) -> Tensor:
    """Return per-image channel correlations normalized by channels * height * width."""
    if features.ndim != 4 or min(features.shape) < 1:
        raise ValueError("Gram features must be a nonempty N x C x H x W tensor")
    batch, channels, height, width = features.shape
    flat = features.reshape(batch, channels, height * width)
    return torch.bmm(flat, flat.transpose(1, 2)) / (channels * height * width)


def content_loss(generated: Tensor, target: Tensor) -> Tensor:
    """Measure mean squared spatial feature differences without broadcasting."""
    if generated.shape != target.shape:
        raise ValueError("Content feature shapes must match")
    return F.mse_loss(generated, target)


def style_loss(generated: Tensor, target_gram: Tensor) -> Tensor:
    """Match normalized channel correlations, permitting different spatial sizes."""
    gram = gram_matrix(generated)
    if gram.shape != target_gram.shape:
        raise ValueError("Style Gram shapes must match")
    return F.mse_loss(gram, target_gram)


def total_variation_loss(pixels: Tensor) -> Tensor:
    """Mean squared horizontal plus vertical differences (zero for singleton axes)."""
    if pixels.ndim != 4 or min(pixels.shape) < 1:
        raise ValueError("TV input must be a nonempty N x C x H x W tensor")
    loss = pixels.sum() * 0.0
    if pixels.shape[-1] > 1:
        loss = loss + (pixels[..., 1:] - pixels[..., :-1]).square().mean()
    if pixels.shape[-2] > 1:
        loss = loss + (pixels[..., 1:, :] - pixels[..., :-1, :]).square().mean()
    return loss
