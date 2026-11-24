"""Experimental FiLM-style reference conditioning for the guided decoder."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from moodshift.engine.guided_decoder import (
    BASELINE_WEIGHTS,
    GuidedLossWeights,
    edge_loss,
)
from moodshift.engine.training import (
    EncoderAdapter,
    _validate_tensor,
    align_features,
    feature_statistics,
)


class StyleConditionedDecoder(nn.Module):
    """Predict a bounded RGB residual using explicit reference-feature conditioning."""

    def __init__(self) -> None:
        """Build a compact CNN and reference-conditioned modulation network."""
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(384, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(35, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 3, 3, padding=1)
        )
        self.condition = nn.Sequential(nn.Linear(768, 64), nn.ReLU(), nn.Linear(64, 67))

    def forward(
        self, aligned: Tensor, content: Tensor, style_features: Tensor
    ) -> Tensor:
        """Modulate decoder activations and color shift using reference statistics."""
        _validate_tensor(aligned, 384, "aligned features")
        _validate_tensor(style_features, 384, "style features")
        _validate_tensor(content, 3, "content")
        if (
            aligned.shape[0] != content.shape[0]
            or style_features.shape[0] != content.shape[0]
            or aligned.device != content.device
            or style_features.device != content.device
            or aligned.dtype != content.dtype
            or style_features.dtype != content.dtype
        ):
            raise ValueError("inputs must have matching batch, device, and dtype")
        if (content < 0).any() or (content > 1).any():
            raise ValueError("content must be in [0, 1]")
        mean, std = feature_statistics(style_features)
        # asinh compresses large activations and supports negative values.
        descriptor = torch.cat(
            (torch.asinh(mean.flatten(1) / 32), torch.asinh(std.flatten(1) / 32)), dim=1
        )
        condition = self.condition(descriptor)
        gamma, beta, color = torch.split(condition, (32, 32, 3), dim=1)
        value = self.feature(aligned / 32.0)
        value = value * (1 + 0.5 * torch.tanh(gamma[:, :, None, None]))
        value = value + 0.5 * torch.tanh(beta[:, :, None, None])
        value = F.interpolate(
            value, size=content.shape[-2:], mode="bilinear", align_corners=False
        )
        local = self.refine(torch.cat((value, content), dim=1))
        shift = 2.0 * torch.tanh(local + 0.5 * torch.tanh(color[:, :, None, None]))
        output = torch.sigmoid(torch.logit(content.clamp(1e-4, 1 - 1e-4)) + shift)
        if not torch.isfinite(output).all():
            raise ValueError("conditioned decoder output is not finite")
        return output


def conditioned_objective(
    encoder: EncoderAdapter,
    decoder: StyleConditionedDecoder,
    content: Tensor,
    style: Tensor,
    weights: GuidedLossWeights = BASELINE_WEIGHTS,
) -> dict[str, Tensor]:
    """Use the guided baseline's losses with explicit reference conditioning."""
    with torch.no_grad():
        content_features = encoder(content)
        style_features = encoder(style)
        target = align_features(content_features, style_features)
        sm, ss = feature_statistics(style_features)
    output = decoder(target, content, style_features)
    features = encoder(output)
    om, os = feature_statistics(features)
    content_loss = F.mse_loss(features, target) / target.square().mean().clamp_min(1e-5)
    style_loss = (F.mse_loss(om, sm) + F.mse_loss(os, ss)) / (
        sm.square().mean() + ss.square().mean()
    ).clamp_min(1e-5)
    identity = decoder(content_features, content, content_features)
    reconstruction = F.mse_loss(identity, content)
    edges = edge_loss(output, content)
    total = (
        weights.content * content_loss
        + weights.style * style_loss
        + weights.identity * reconstruction
        + weights.edge * edges
    )
    return {
        "loss": total,
        "content_loss": content_loss,
        "style_loss": style_loss,
        "reconstruction_loss": reconstruction,
        "edge_loss": edges,
    }


def conditioned_step(
    encoder: EncoderAdapter,
    decoder: StyleConditionedDecoder,
    optimizer: torch.optim.Optimizer,
    content: Tensor,
    style: Tensor,
    weights: GuidedLossWeights = BASELINE_WEIGHTS,
) -> dict[str, float]:
    """Perform one finite, clipped decoder-only Adam-compatible update."""
    decoder_ids = {id(p) for p in decoder.parameters() if p.requires_grad}
    optimizer_ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
    if len(optimizer_ids) != len(decoder_ids) or set(optimizer_ids) != decoder_ids:
        raise ValueError("optimizer must contain exactly the decoder parameters")
    optimizer.zero_grad(set_to_none=True)
    losses = conditioned_objective(encoder, decoder, content, style, weights)
    if not all(torch.isfinite(value) for value in losses.values()):
        raise ValueError("nonfinite training loss")
    losses["loss"].backward()
    nn.utils.clip_grad_norm_(decoder.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return {name: value.item() for name, value in losses.items()}
