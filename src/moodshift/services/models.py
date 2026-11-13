"""Validated mood requests and truthful result metadata, without importing PyTorch."""

import math
from dataclasses import dataclass
from typing import Literal

from PIL import Image

from moodshift.engine.settings import TransferSettings

type Mode = Literal["grade", "neural", "hybrid"]
type Quality = Literal["preview", "balanced", "high"]


@dataclass(frozen=True, slots=True)
class MoodSettings:
    """Final blend intensity (0..100) and bounded neural processing controls."""

    mode: Mode = "grade"
    intensity: float = 70.0
    quality: Quality = "balanced"
    content_preservation: float = 0.7
    max_size: int | None = None
    steps: int | None = None
    seed: int = 0
    grain_enabled: bool = True

    def __post_init__(self) -> None:
        """Reject invalid modes, nonfinite intensity, and unsafe neural controls."""
        if self.mode not in ("grade", "neural", "hybrid"):
            raise ValueError("mode must be grade, neural, or hybrid")
        if self.quality not in ("preview", "balanced", "high"):
            raise ValueError("quality must be preview, balanced, or high")
        if (
            isinstance(self.intensity, bool)
            or not isinstance(self.intensity, (int, float))
            or not math.isfinite(self.intensity)
            or not 0 <= self.intensity <= 100
        ):
            raise ValueError("intensity must be finite and in [0,100]")
        if type(self.grain_enabled) is not bool:
            raise ValueError("grain_enabled must be boolean")
        self.neural_settings()  # Reuse the engine's existing bounds/validation.

    def neural_settings(self) -> TransferSettings:
        """Resolve service quality independently of the unchanged engine defaults."""
        steps, max_size = {
            "preview": (20, 128),
            "balanced": (60, 256),
            "high": (120, 384),
        }[self.quality]
        return TransferSettings(
            style_strength=1.0,
            content_preservation=self.content_preservation,
            steps=steps if self.steps is None else self.steps,
            max_size=max_size if self.max_size is None else self.max_size,
            seed=self.seed,
        )


@dataclass(frozen=True, slots=True)
class EffectiveSettings:
    """Controls actually used, distinguishing requested quality from executed steps."""

    intensity: float
    quality: Quality
    content_preservation: float | None
    neural_steps: int
    neural_max_size: int | None
    output_max_size: int | None
    seed: int
    grain_enabled: bool


@dataclass(frozen=True, slots=True)
class MoodMetadata:
    """Execution facts; reference source is none unless neural inference consumed it."""

    preset_id: str | None
    mode: Mode
    actual_engine: str
    device: str | None
    neural_applied: bool
    grading_applied: bool
    reference_source: Literal["none", "synthetic-preset", "user"]
    original_size: tuple[int, int]
    output_size: tuple[int, int]
    neural_size: tuple[int, int] | None
    settings: EffectiveSettings


@dataclass(frozen=True, slots=True)
class MoodResult:
    """Independent output RGB pixels and immutable metadata about their processing."""

    image: Image.Image
    metadata: MoodMetadata
