"""Typed presentation inputs and outputs, independent of Gradio components."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from moodshift.services.models import MoodSettings


@dataclass(frozen=True, slots=True)
class UiParameters:
    """One snapshot of form controls; hidden controls do not affect grade mode."""

    content: str | None
    mode: str = "grade"
    preset: str = "cinematic"
    reference: str | None = None
    intensity: float = 70
    quality: str = "preview"
    preservation: float = 0.7
    custom_steps: bool = False
    steps: int = 20
    limit_output: bool = False
    max_size: int = 128
    offline: bool = True
    device: str = "auto"
    seed: int = 0
    grain: bool = True
    style_strength: float = 1.0
    boost_enabled: bool = False
    boost_reference: str | None = None
    boost_strength: float = 50

    def settings(self) -> MoodSettings:
        """Validate actual controls through the existing service settings model."""
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("Choose Auto, CPU, or CUDA for the device.")
        for value in (
            self.custom_steps,
            self.limit_output,
            self.offline,
            self.grain,
            self.boost_enabled,
        ):
            if type(value) is not bool:
                raise ValueError("Checkbox controls must be true or false.")
        if (
            isinstance(self.boost_strength, bool)
            or not isinstance(self.boost_strength, (int, float))
            or not math.isfinite(self.boost_strength)
            or not 0 <= self.boost_strength <= 100
        ):
            raise ValueError("Reference Mood Boost strength must be between 0 and 100.")
        return MoodSettings(
            mode=self.mode,
            intensity=self.intensity,
            quality=self.quality if self.mode != "grade" else "preview",
            content_preservation=self.preservation if self.mode != "grade" else 0.7,
            steps=self.steps if self.custom_steps and self.mode != "grade" else None,
            max_size=self.max_size if self.limit_output else None,
            seed=self.seed,
            grain_enabled=self.grain,
            neural_style_strength=self.style_strength if self.mode != "grade" else 1.0,
        )

    def selection(self) -> tuple[str | None, str | None]:
        """Route external neural references without sending an unused preset."""
        if self.mode == "grade":
            return self.preset, None
        if self.mode == "neural" and self.reference:
            return None, self.reference
        return self.preset, self.reference


@dataclass(frozen=True, slots=True)
class RenderedResult:
    """Display-ready images, local PNG download, and actual execution facts."""

    image: Image.Image
    comparison: tuple[Image.Image, Image.Image]
    download: Path
    summary: str
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ModePresentation:
    """Mode-dependent visibility and explanatory text for the form."""

    neural_visible: bool
    preset_enabled: bool
    grain_enabled: bool
    explanation: str
