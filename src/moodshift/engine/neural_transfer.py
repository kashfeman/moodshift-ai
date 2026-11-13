"""Reference-guided optimization of RGB pixels through a frozen pretrained CNN."""

from pathlib import Path

import torch
from PIL import Image
from torch import Tensor

from moodshift.config import DEFAULT_CONFIG, AppConfig
from moodshift.engine.exceptions import NeuralTransferError
from moodshift.engine.features import CONTENT_LAYER, STYLE_LAYERS, FeatureExtractor
from moodshift.engine.losses import (
    content_loss,
    gram_matrix,
    style_loss,
    total_variation_loss,
)
from moodshift.engine.model_loader import load_feature_extractor, select_device
from moodshift.engine.settings import TransferSettings
from moodshift.engine.tensors import image_to_tensor, tensor_to_image
from moodshift.imaging.io import load_image
from moodshift.imaging.transforms import resize_to_fit


class NeuralStyleTransfer:
    """Reuse one lazy pretrained extractor across sequential transfer requests.

    Only the generated pixels are optimized. An injected extractor is intended
    for isolated tests; production defaults always load official pretrained CNN
    features. Device errors are reported; no fake or partial output is returned.
    """

    def __init__(
        self,
        *,
        device: str = "auto",
        config: AppConfig = DEFAULT_CONFIG,
        allow_download: bool = True,
        extractor: FeatureExtractor | None = None,
    ) -> None:
        """Select a device without downloading weights until a nonzero transfer."""
        self.device = select_device(device)
        self.config = config
        self.allow_download = allow_download
        self._extractor = extractor

    def transfer(
        self,
        content_path: str | Path,
        style_path: str | Path,
        settings: TransferSettings | None = None,
    ) -> Image.Image:
        """Validate both files and generate RGB pixels at content processing size.

        Zero style returns a separate full-resolution decoded content image,
        bypassing resize, minimum CNN dimensions, and all model loading.
        """
        settings = TransferSettings() if settings is None else settings
        try:
            content = load_image(content_path, self.config)
            style = load_image(style_path, self.config)
            if settings.style_strength == 0:
                return content.copy()
            content = resize_to_fit(content, (settings.max_size, settings.max_size))
            style = resize_to_fit(style, (settings.max_size, settings.max_size))
            if min(*content.size, *style.size) < 17:
                raise ValueError(
                    "Content and style must each be at least 17px per axis after resize"
                )
            if self._extractor is None:
                self._extractor = load_feature_extractor(
                    self.device, allow_download=self.allow_download
                )
            with torch.enable_grad():
                return self._optimize(content, style, settings)
        except NeuralTransferError:
            raise
        except (RuntimeError, MemoryError) as error:
            raise NeuralTransferError(
                "Neural optimization failed; "
                "try CPU, preview quality, or a smaller resolution"
            ) from error

    def _optimize(
        self, content: Image.Image, style: Image.Image, settings: TransferSettings
    ) -> Image.Image:
        extractor = self._extractor
        assert extractor is not None
        content_pixels = image_to_tensor(content, self.device)
        style_pixels = image_to_tensor(style, self.device)
        with torch.no_grad():
            target_content = extractor(content_pixels)[CONTENT_LAYER].detach()
            target_style = {
                name: gram_matrix(value).detach()
                for name, value in extractor(style_pixels).items()
                if name in STYLE_LAYERS
            }
            content_scale = target_content.square().mean().clamp_min(1e-4)
            style_scales = {
                name: value.square().mean().clamp_min(1e-4)
                for name, value in target_style.items()
            }
        if set(target_style) != set(STYLE_LAYERS):
            raise NeuralTransferError(
                "Feature extractor did not return all required style layers"
            )
        generator = torch.Generator(device="cpu").manual_seed(settings.seed)
        noise = torch.randn(content_pixels.shape, generator=generator).to(self.device)
        output = (
            (content_pixels + noise * 0.001).clamp(0, 1).detach().requires_grad_(True)
        )
        optimizer = torch.optim.Adam([output], lr=0.02)
        for _ in range(settings.steps):
            optimizer.zero_grad(set_to_none=True)
            features = extractor(output)
            content_term = (
                content_loss(features[CONTENT_LAYER], target_content) / content_scale
            )
            style_term = torch.stack(
                [
                    style_loss(features[name], target) / style_scales[name]
                    for name, target in target_style.items()
                ]
            ).mean()
            loss = (
                10.0 * settings.content_preservation * content_term
                + 5.0 * settings.style_strength * style_term
                + 0.01 * total_variation_loss(output)
            )
            self._require_finite(loss, "loss")
            loss.backward()
            if output.grad is None:
                raise NeuralTransferError("No gradient reached the generated image")
            self._require_finite(output.grad, "gradient")
            torch.nn.utils.clip_grad_norm_(
                [output], max_norm=10.0, error_if_nonfinite=True
            )
            optimizer.step()
            with torch.no_grad():
                self._require_finite(output, "output")
                output.clamp_(0, 1)
        return tensor_to_image(output)

    @staticmethod
    def _require_finite(value: Tensor, name: str) -> None:
        if not torch.isfinite(value).all():
            raise NeuralTransferError(
                f"Nonfinite neural {name}; no result was returned"
            )
