"""Content-guided experimental decoder and its differentiable training objective."""

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from moodshift.engine.training import (
    EncoderAdapter,
    _validate_tensor,
    align_features,
    feature_statistics,
)


@dataclass(frozen=True, slots=True)
class GuidedLossWeights:
    """Nonnegative finite weights for the four guided training losses."""

    content: float = 1.0
    style: float = 1.0
    identity: float = 10.0
    edge: float = 2.0

    def __post_init__(self) -> None:
        """Reject invalid values and an objective with no active terms."""
        values = (self.content, self.style, self.identity, self.edge)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            for value in values
        ):
            raise ValueError("loss weights must be finite nonnegative numbers")
        if not any(values):
            raise ValueError("at least one loss weight must be positive")


BASELINE_WEIGHTS = GuidedLossWeights()
STRONGER_STYLE_WEIGHTS = GuidedLossWeights(content=1, style=3, identity=5, edge=1)


class GuidedDecoder(nn.Module):
    """Predict bounded style-dependent RGB logit shifts with a content skip."""

    def __init__(self) -> None:
        """Create a small conditioned CNN; weights are trained from scratch."""
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(384, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(35, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1),
        )

    def forward(self, features: Tensor, content: Tensor) -> Tensor:
        """Combine aligned style features and RGB structure into a bounded image."""
        _validate_tensor(features, 384, "features")
        _validate_tensor(content, 3, "content")
        if features.shape[0] != content.shape[0] or features.device != content.device:
            raise ValueError("features and content must share batch size and device")
        if (
            features.dtype != content.dtype
            or (content < 0).any()
            or (content > 1).any()
        ):
            raise ValueError("content must match feature dtype and be in [0, 1]")
        value = self.feature(features / 32.0)
        value = F.interpolate(
            value, size=content.shape[-2:], mode="bilinear", align_corners=False
        )
        shift = 2.0 * torch.tanh(self.refine(torch.cat((value, content), dim=1)))
        # Clamp only for the logit transform, avoiding infinities at black/white.
        logits = torch.logit(content.clamp(1e-4, 1 - 1e-4))
        return torch.sigmoid(logits + shift)


def edge_loss(output: Tensor, content: Tensor) -> Tensor:
    """Measure horizontal and vertical RGB gradient differences."""
    return F.l1_loss(
        output[..., 1:] - output[..., :-1], content[..., 1:] - content[..., :-1]
    ) + F.l1_loss(
        output[..., 1:, :] - output[..., :-1, :],
        content[..., 1:, :] - content[..., :-1, :],
    )


def guided_objective(
    encoder: EncoderAdapter,
    decoder: GuidedDecoder,
    content: Tensor,
    style: Tensor,
    weights: GuidedLossWeights = BASELINE_WEIGHTS,
) -> dict[str, Tensor]:
    """Match style features while preserving RGB identity and content gradients."""
    with torch.no_grad():
        content_features = encoder(content)
        style_features = encoder(style)
        target = align_features(content_features, style_features)
        sm, ss = feature_statistics(style_features)
    output = decoder(target, content)
    output_features = encoder(output)
    om, os = feature_statistics(output_features)
    content_loss = F.mse_loss(
        output_features, target
    ) / target.square().mean().clamp_min(1e-5)
    style_loss = (F.mse_loss(om, sm) + F.mse_loss(os, ss)) / (
        sm.square().mean() + ss.square().mean()
    ).clamp_min(1e-5)
    identity = decoder(content_features, content)
    reconstruction_loss = F.mse_loss(identity, content)
    edges = edge_loss(output, content)
    total = (
        weights.content * content_loss
        + weights.style * style_loss
        + weights.identity * reconstruction_loss
        + weights.edge * edges
    )
    return {
        "loss": total,
        "content_loss": content_loss,
        "style_loss": style_loss,
        "reconstruction_loss": reconstruction_loss,
        "edge_loss": edges,
    }


def guided_step(
    encoder: EncoderAdapter,
    decoder: GuidedDecoder,
    optimizer: torch.optim.Optimizer,
    content: Tensor,
    style: Tensor,
    weights: GuidedLossWeights = BASELINE_WEIGHTS,
) -> dict[str, float]:
    """Update decoder parameters once with finite checks and gradient clipping."""
    decoder_ids = {id(p) for p in decoder.parameters() if p.requires_grad}
    optimizer_ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
    if len(optimizer_ids) != len(decoder_ids) or set(optimizer_ids) != decoder_ids:
        raise ValueError("optimizer must contain exactly the decoder parameters")
    optimizer.zero_grad(set_to_none=True)
    losses = guided_objective(encoder, decoder, content, style, weights)
    if not all(torch.isfinite(v) for v in losses.values()):
        raise ValueError("nonfinite training loss")
    losses["loss"].backward()
    nn.utils.clip_grad_norm_(decoder.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return {name: value.item() for name, value in losses.items()}
