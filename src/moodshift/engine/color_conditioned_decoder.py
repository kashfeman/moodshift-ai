"""RGB-statistics-conditioned experimental version of the guided decoder."""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from moodshift.engine.guided_decoder import (
    BASELINE_WEIGHTS,
    GuidedLossWeights,
    edge_loss,
)
from moodshift.engine.style_conditioned_decoder import StyleConditionedDecoder
from moodshift.engine.training import (
    EncoderAdapter,
    _validate_tensor,
    align_features,
    feature_statistics,
)


class ColorConditionedDecoder(nn.Module):
    """Extend the existing feature-conditioned decoder with learned RGB statistics."""

    def __init__(self) -> None:
        """Initialize the original feature conditioner before the new RGB branch."""
        super().__init__()
        self.base = StyleConditionedDecoder()
        self.rgb_condition = nn.Sequential(
            nn.Linear(6, 16), nn.ReLU(), nn.Linear(16, 3)
        )

    def forward(
        self,
        aligned: Tensor,
        content: Tensor,
        style_features: Tensor,
        style_rgb: Tensor,
    ) -> Tensor:
        """Predict a bounded RGB residual from reference means and deviations."""
        _validate_tensor(style_rgb, 3, "style RGB")
        if (
            style_rgb.shape[0] != content.shape[0]
            or style_rgb.device != content.device
            or style_rgb.dtype != content.dtype
        ):
            raise ValueError("style RGB must match content batch, device, and dtype")
        if (style_rgb < 0).any() or (style_rgb > 1).any():
            raise ValueError("style RGB must be in [0, 1]")
        mean, std = feature_statistics(style_rgb)
        descriptor = torch.cat((mean.flatten(1), std.flatten(1)), dim=1)
        color_shift = 0.5 * torch.tanh(self.rgb_condition(descriptor))[:, :, None, None]
        base = self.base(aligned, content, style_features)
        result = torch.sigmoid(torch.logit(base.clamp(1e-6, 1 - 1e-6)) + color_shift)
        if not torch.isfinite(result).all():
            raise ValueError("RGB-conditioned decoder output is not finite")
        return result


def rgb_statistics_loss(generated: Tensor, reference: Tensor) -> Tensor:
    """Match per-image RGB means and population standard deviations in [0, 1]."""
    _validate_tensor(generated, 3, "generated RGB")
    _validate_tensor(reference, 3, "reference RGB")
    if (
        generated.shape[0] != reference.shape[0]
        or generated.device != reference.device
        or generated.dtype != reference.dtype
    ):
        raise ValueError("RGB tensors must have matching batch, device, and dtype")
    if (
        (generated < 0).any()
        or (generated > 1).any()
        or (reference < 0).any()
        or (reference > 1).any()
    ):
        raise ValueError("RGB tensors must be in [0, 1]")
    generated_mean, generated_std = feature_statistics(generated)
    reference_mean, reference_std = feature_statistics(reference)
    return F.mse_loss(generated_mean, reference_mean) + F.mse_loss(
        generated_std, reference_std
    )


def color_objective(
    encoder: EncoderAdapter,
    decoder: ColorConditionedDecoder,
    content: Tensor,
    style: Tensor,
    weights: GuidedLossWeights = BASELINE_WEIGHTS,
    *,
    rgb_weight: float = 0.0,
    paired_style: Tensor | None = None,
    paired_weight: float = 0.0,
    response_target: float = 0.25,
) -> dict[str, Tensor]:
    """Apply the existing guided loss definitions to the RGB-conditioned model."""
    if (
        isinstance(rgb_weight, bool)
        or not isinstance(rgb_weight, (int, float))
        or not math.isfinite(rgb_weight)
        or rgb_weight < 0
    ):
        raise ValueError("rgb_weight must be finite and nonnegative")
    for name, value in (
        ("paired_weight", paired_weight),
        ("response_target", response_target),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"{name} must be finite and nonnegative")
    if response_target > 1:
        raise ValueError("response_target must be between 0 and 1")
    if paired_weight and paired_style is None:
        raise ValueError("paired_style is required when paired_weight is positive")
    with torch.no_grad():
        cf = encoder(content)
        sf = encoder(style)
        target = align_features(cf, sf)
        sm, ss = feature_statistics(sf)
    output = decoder(target, content, sf, style)
    of = encoder(output)
    om, os = feature_statistics(of)
    content_loss = F.mse_loss(of, target) / target.square().mean().clamp_min(1e-5)
    style_loss = (F.mse_loss(om, sm) + F.mse_loss(os, ss)) / (
        sm.square().mean() + ss.square().mean()
    ).clamp_min(1e-5)
    identity = decoder(cf, content, cf, content)
    reconstruction = F.mse_loss(identity, content)
    edges = edge_loss(output, content)
    total = (
        weights.content * content_loss
        + weights.style * style_loss
        + weights.identity * reconstruction
        + weights.edge * edges
    )
    result = {
        "loss": total,
        "content_loss": content_loss,
        "style_loss": style_loss,
        "reconstruction_loss": reconstruction,
        "edge_loss": edges,
    }
    if rgb_weight:
        rgb_loss = rgb_statistics_loss(output, style)
        result["rgb_color_loss"] = rgb_loss
        result["loss"] = total + rgb_weight * rgb_loss
    if paired_weight:
        assert paired_style is not None
        with torch.no_grad():
            second_sf = encoder(paired_style)
            second_target = align_features(cf, second_sf)
        second_output = decoder(second_target, content, second_sf, paired_style)
        output_delta = output.mean(dim=(-2, -1)) - second_output.mean(dim=(-2, -1))
        reference_delta = style.mean(dim=(-2, -1)) - paired_style.mean(dim=(-2, -1))
        paired_loss = F.mse_loss(output_delta, response_target * reference_delta)
        result["paired_color_response_loss"] = paired_loss
        result["loss"] = result["loss"] + paired_weight * paired_loss
    return result


def color_matching_objective(
    encoder: EncoderAdapter,
    decoder: ColorConditionedDecoder,
    content: Tensor,
    style: Tensor,
    weights: GuidedLossWeights = BASELINE_WEIGHTS,
) -> dict[str, Tensor]:
    """Add a conservative weight-1 RGB-statistics term to the existing losses."""
    return color_objective(encoder, decoder, content, style, weights, rgb_weight=1.0)


def color_step(
    encoder: EncoderAdapter,
    decoder: ColorConditionedDecoder,
    optimizer: torch.optim.Optimizer,
    content: Tensor,
    style: Tensor,
    weights: GuidedLossWeights = BASELINE_WEIGHTS,
    *,
    rgb_weight: float = 0.0,
    paired_style: Tensor | None = None,
    paired_weight: float = 0.0,
    response_target: float = 0.25,
) -> dict[str, float]:
    """Perform one clipped, decoder-only update using finite losses."""
    ids = {id(p) for p in decoder.parameters() if p.requires_grad}
    optimizer_ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
    if len(optimizer_ids) != len(ids) or set(optimizer_ids) != ids:
        raise ValueError("optimizer must contain exactly the decoder parameters")
    optimizer.zero_grad(set_to_none=True)
    losses = color_objective(
        encoder,
        decoder,
        content,
        style,
        weights,
        rgb_weight=rgb_weight,
        paired_style=paired_style,
        paired_weight=paired_weight,
        response_target=response_target,
    )
    if not all(torch.isfinite(v) for v in losses.values()):
        raise ValueError("nonfinite training loss")
    losses["loss"].backward()
    nn.utils.clip_grad_norm_(decoder.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return {name: value.item() for name, value in losses.items()}


def color_matching_step(
    encoder: EncoderAdapter,
    decoder: ColorConditionedDecoder,
    optimizer: torch.optim.Optimizer,
    content: Tensor,
    style: Tensor,
    weights: GuidedLossWeights = BASELINE_WEIGHTS,
) -> dict[str, float]:
    """Update the decoder using the separate reference-color objective."""
    return color_step(
        encoder, decoder, optimizer, content, style, weights, rgb_weight=1.0
    )
