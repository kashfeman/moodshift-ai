"""Application orchestration for photographic grading and real neural transfer."""

from pathlib import Path
from typing import Protocol

from PIL import Image

from moodshift.config import DEFAULT_CONFIG, AppConfig
from moodshift.engine.settings import TransferSettings
from moodshift.imaging.grading import apply_grade, blend_images
from moodshift.imaging.io import load_image
from moodshift.imaging.transforms import resize_to_fit
from moodshift.presets import MoodPreset
from moodshift.services.models import (
    EffectiveSettings,
    MoodMetadata,
    MoodResult,
    MoodSettings,
)
from moodshift.services.references import reference_path, resolve_preset


class NeuralBackend(Protocol):
    """Existing engine boundary; metadata explicitly labels injected backends."""

    @property
    def device(self) -> object:
        """Report the actual execution device."""
        ...

    def transfer(
        self,
        content_path: str | Path,
        style_path: str | Path,
        settings: TransferSettings | None = None,
    ) -> Image.Image:
        """Generate RGB pixels using the selected reference."""
        ...


class MoodTransferService:
    """Reuse a lazy neural backend; grading-only requests never import PyTorch."""

    def __init__(
        self,
        *,
        config: AppConfig = DEFAULT_CONFIG,
        device: str = "auto",
        allow_download: bool = True,
        neural_engine: NeuralBackend | None = None,
    ) -> None:
        """Keep device/model initialization deferred until an actual neural request."""
        if device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu, or cuda")
        self.config = config
        self.device = device
        self.allow_download = allow_download
        self._engine = neural_engine
        self._engine_name = (
            "injected-neural-backend" if neural_engine else "squeezenet1_1"
        )

    def transfer(
        self,
        content_path: str | Path,
        *,
        preset_id: str | None = None,
        reference: str | Path | None = None,
        settings: MoodSettings | None = None,
    ) -> MoodResult:
        """Grade, generate neural output, or combine both, then blend intensity once.

        Default output has original dimensions. Explicit max_size limits output;
        otherwise low-resolution neural pixels are restored to the original grid.
        No crop or geometric warp is applied. Zero intensity bypasses all effects.
        """
        settings = MoodSettings() if settings is None else settings
        preset = resolve_preset(preset_id, reference, settings.mode)
        content = load_image(content_path, self.config)
        user_reference = Path(reference) if reference is not None else None
        if user_reference is not None:
            load_image(user_reference, self.config).close()
        original_size = content.size
        active = settings.intensity > 0
        if active and settings.max_size is not None:
            content = resize_to_fit(content, (settings.max_size, settings.max_size))
        neural = active and settings.mode in ("neural", "hybrid")
        grading = active and settings.mode in ("grade", "hybrid")
        neural_size = None
        stylized = content.copy()
        if neural:
            engine = self._get_engine()
            with reference_path(user_reference, preset) as style_path:
                stylized = engine.transfer(
                    content_path, style_path, settings.neural_settings()
                )
            neural_size = stylized.size
            if stylized.size != content.size:
                stylized = stylized.resize(content.size, Image.Resampling.LANCZOS)
        if grading:
            assert preset is not None
            stylized = apply_grade(
                stylized,
                preset.grading,
                seed=settings.seed,
                grain_enabled=settings.grain_enabled,
            )
        output = (
            blend_images(content, stylized, settings.intensity / 100)
            if active
            else content.copy()
        )
        metadata = self._metadata(
            settings,
            preset,
            original_size,
            output.size,
            neural_size,
            user_reference is not None,
        )
        return MoodResult(output, metadata)

    def _metadata(
        self,
        settings: MoodSettings,
        preset: MoodPreset | None,
        original_size: tuple[int, int],
        output_size: tuple[int, int],
        neural_size: tuple[int, int] | None,
        user_reference: bool,
    ) -> MoodMetadata:
        """Describe operations actually executed, not just requested controls."""
        neural = neural_size is not None
        grading = settings.intensity > 0 and settings.mode in ("grade", "hybrid")
        effective_neural = settings.neural_settings()
        return MoodMetadata(
            preset_id=preset.id if preset is not None else None,
            mode=settings.mode,
            actual_engine=self._engine_name
            if neural
            else "numpy-opencv"
            if grading
            else "none",
            device=str(self._engine.device) if neural else "cpu" if grading else None,
            neural_applied=neural,
            grading_applied=grading,
            reference_source="user"
            if neural and user_reference
            else "synthetic-preset"
            if neural
            else "none",
            original_size=original_size,
            output_size=output_size,
            neural_size=neural_size,
            settings=EffectiveSettings(
                intensity=settings.intensity,
                quality=settings.quality,
                content_preservation=settings.content_preservation if neural else None,
                neural_steps=effective_neural.steps if neural else 0,
                neural_max_size=effective_neural.max_size if neural else None,
                output_max_size=settings.max_size if settings.intensity > 0 else None,
                seed=settings.seed,
                grain_enabled=grading
                and preset is not None
                and settings.grain_enabled
                and preset.grading.grain > 0,
            ),
        )

    def _get_engine(self) -> NeuralBackend:
        if self._engine is None:
            from moodshift.engine.neural_transfer import NeuralStyleTransfer

            self._engine = NeuralStyleTransfer(
                config=self.config,
                device=self.device,
                allow_download=self.allow_download,
            )
        return self._engine
