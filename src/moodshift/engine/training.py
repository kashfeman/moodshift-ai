"""Trainable decoder foundation using the existing SqueezeNet feature prefix.

Layer 9 (Fire6) has 384 channels and approximately one sixteenth of the input
resolution. Exact output size is supplied explicitly because pooling rounds.
"""

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from moodshift.engine.features import SqueezeNetFeatures


def _validate_tensor(value: Tensor, channels: int, name: str) -> None:
    if (
        not isinstance(value, Tensor)
        or value.ndim != 4
        or min(value.shape) < 1
        or value.shape[1] != channels
        or not value.is_floating_point()
        or not torch.isfinite(value).all()
    ):
        raise ValueError(
            f"{name} must be a finite floating N x {channels} x H x W tensor"
        )


def feature_statistics(features: Tensor, eps: float = 1e-5) -> tuple[Tensor, Tensor]:
    """Return per-sample channel mean and stabilized population standard deviation."""
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if (
        not isinstance(features, Tensor)
        or features.ndim != 4
        or min(features.shape) < 1
    ):
        raise ValueError("features must be a nonempty N x C x H x W tensor")
    _validate_tensor(features, features.shape[1], "features")
    mean = features.mean(dim=(2, 3), keepdim=True)
    std = (features.var(dim=(2, 3), unbiased=False, keepdim=True) + eps).sqrt()
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise ValueError("feature statistics are not finite")
    return mean, std


def align_features(content: Tensor, style: Tensor, eps: float = 1e-5) -> Tensor:
    """Align content channel statistics to style, retaining content spatial size."""
    if not isinstance(content, Tensor) or content.ndim != 4 or min(content.shape) < 1:
        raise ValueError("content must be a nonempty N x C x H x W tensor")
    _validate_tensor(content, content.shape[1], "content")
    _validate_tensor(style, content.shape[1], "style")
    if (
        content.shape[0] != style.shape[0]
        or content.device != style.device
        or content.dtype != style.dtype
    ):
        raise ValueError(
            "content and style must have matching batch, device, and dtype"
        )
    content_mean, content_std = feature_statistics(content, eps)
    style_mean, style_std = feature_statistics(style, eps)
    aligned = (content - content_mean) / content_std * style_std + style_mean
    if not torch.isfinite(aligned).all():
        raise ValueError("aligned features are not finite")
    return aligned


class EncoderAdapter(nn.Module):
    """Select the existing frozen SqueezeNet layer 9, preserving input gradients."""

    channels = 384
    layer = "9"

    def __init__(self, encoder: SqueezeNetFeatures) -> None:
        """Wrap an existing SqueezeNetFeatures instance without loading weights."""
        super().__init__()
        self.encoder = encoder
        self.encoder.requires_grad_(False)
        self.encoder.eval()

    def train(self, mode: bool = True) -> "EncoderAdapter":
        """Keep the frozen encoder in evaluation mode."""
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, pixels: Tensor) -> Tensor:
        """Encode RGB pixels in [0, 1], with each spatial axis at least 17."""
        _validate_tensor(pixels, 3, "pixels")
        if min(pixels.shape[-2:]) < 17 or (pixels < 0).any() or (pixels > 1).any():
            raise ValueError(
                "pixels must be in [0, 1] with spatial dimensions at least 17"
            )
        features = self.encoder(pixels)[self.layer]
        _validate_tensor(features, self.channels, "encoder features")
        return features


class ImageDecoder(nn.Module):
    """Upsample layer-9 features to an RGB image at the requested exact size."""

    def __init__(self) -> None:
        """Build a compact convolutional decoder with a bounded RGB output."""
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(nn.Conv2d(384, 128, 3, padding=1), nn.ReLU()),
                nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU()),
                nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.ReLU()),
            ]
        )
        self.output = nn.Conv2d(32, 3, 3, padding=1)

    def forward(self, features: Tensor, output_size: tuple[int, int]) -> Tensor:
        """Decode features with interpolation and return RGB values in [0, 1]."""
        _validate_tensor(features, 384, "features")
        if (
            not isinstance(output_size, tuple)
            or len(output_size) != 2
            or any(type(size) is not int or size < 1 for size in output_size)
        ):
            raise ValueError("output_size must be a pair of positive integers")
        value = features
        for block in self.blocks:
            value = block(value)
            value = F.interpolate(
                value, scale_factor=2, mode="bilinear", align_corners=False
            )
        value = F.interpolate(
            value, size=output_size, mode="bilinear", align_corners=False
        )
        result = torch.sigmoid(self.output(value))
        if not torch.isfinite(result).all():
            raise ValueError("decoder output is not finite")
        return result


@dataclass(frozen=True)
class TrainingConfig:
    """Adam and objective settings for a single decoder training step."""

    learning_rate: float = 1e-4
    content_weight: float = 1.0
    style_weight: float = 1.0
    reconstruction_weight: float = 10.0
    eps: float = 1e-5

    def __post_init__(self) -> None:
        """Reject invalid optimization settings."""
        for name in (
            "learning_rate",
            "content_weight",
            "style_weight",
            "reconstruction_weight",
            "eps",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite")
            if value < 0 or (name in ("learning_rate", "eps") and value == 0):
                raise ValueError(
                    f"{name} must be positive or a nonnegative loss weight"
                )
        if (
            self.content_weight == 0
            and self.style_weight == 0
            and self.reconstruction_weight == 0
        ):
            raise ValueError("at least one loss weight must be positive")


def make_optimizer(decoder: ImageDecoder, config: TrainingConfig) -> torch.optim.Adam:
    """Create an Adam optimizer containing only decoder parameters."""
    return torch.optim.Adam(decoder.parameters(), lr=config.learning_rate)


def decoder_objective(
    encoder: EncoderAdapter,
    decoder: ImageDecoder,
    content: Tensor,
    style: Tensor,
    config: TrainingConfig,
) -> dict[str, Tensor]:
    """Compute normalized feature losses and an RGB identity reconstruction loss."""
    with torch.no_grad():
        content_features = encoder(content)
        style_features = encoder(style)
        target = align_features(content_features, style_features, config.eps)
        style_mean, style_std = feature_statistics(style_features, config.eps)
    generated = decoder(target, tuple(content.shape[-2:]))
    generated_features = encoder(generated)
    content_loss = F.mse_loss(
        generated_features, target
    ) / target.square().mean().clamp_min(config.eps)
    generated_mean, generated_std = feature_statistics(generated_features, config.eps)
    style_loss = (
        F.mse_loss(generated_mean, style_mean) + F.mse_loss(generated_std, style_std)
    ) / (style_mean.square().mean() + style_std.square().mean()).clamp_min(config.eps)
    reconstruction = decoder(content_features, tuple(content.shape[-2:]))
    reconstruction_loss = F.mse_loss(reconstruction, content)
    loss = (
        config.content_weight * content_loss
        + config.style_weight * style_loss
        + config.reconstruction_weight * reconstruction_loss
    )
    return {
        "loss": loss,
        "content_loss": content_loss,
        "style_loss": style_loss,
        "reconstruction_loss": reconstruction_loss,
    }


def training_step(
    encoder: EncoderAdapter,
    decoder: ImageDecoder,
    optimizer: torch.optim.Optimizer,
    content: Tensor,
    style: Tensor,
    config: TrainingConfig,
) -> dict[str, float]:
    """Perform one decoder update using feature reconstruction and style statistics.

    Targets are detached; the generated image is re-encoded with autograd enabled
    so the frozen encoder transmits gradients to the decoder.
    """
    decoder_ids = {id(p) for p in decoder.parameters() if p.requires_grad}
    optimizer_ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
    if len(optimizer_ids) != len(decoder_ids) or set(optimizer_ids) != decoder_ids:
        raise ValueError(
            "optimizer must contain exactly the trainable decoder parameters"
        )
    optimizer.zero_grad(set_to_none=True)
    losses = decoder_objective(encoder, decoder, content, style, config)
    loss = losses["loss"]
    if not torch.isfinite(loss):
        raise ValueError("training loss is not finite")
    loss.backward()
    if any(
        p.grad is not None and not torch.isfinite(p.grad).all()
        for p in decoder.parameters()
    ):
        optimizer.zero_grad(set_to_none=True)
        raise ValueError("decoder gradients are not finite")
    nn.utils.clip_grad_norm_(
        decoder.parameters(), max_norm=1.0, error_if_nonfinite=True
    )
    optimizer.step()
    return {name: value.item() for name, value in losses.items()}
