"""Immutable photographic grade parameters independent of the preset registry."""

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GradeParameters:
    """CIELAB a/b offsets and bounded tone, chroma, bloom, and grain controls.

    Gamma above one brightens; contrast above one deepens the tone curve.
    Black lift, bloom, and grain are fractions of normalized RGB/luminance.
    """

    shadows: tuple[float, float] = (0.0, 0.0)
    midtones: tuple[float, float] = (0.0, 0.0)
    highlights: tuple[float, float] = (0.0, 0.0)
    contrast: float = 1.0
    gamma: float = 1.0
    saturation: float = 1.0
    vibrance: float = 0.0
    black_lift: float = 0.0
    white_point: float = 1.0
    bloom: float = 0.0
    grain: float = 0.0

    def __post_init__(self) -> None:
        """Reject extreme, nonfinite, or mutable processing controls."""
        for name in ("shadows", "midtones", "highlights"):
            pair = getattr(self, name)
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError(f"{name} must be an immutable a/b pair")
            for value in pair:
                _validate_number(name, value, -20, 20)
        for name, low, high in (
            ("contrast", 0.7, 1.4),
            ("gamma", 0.7, 1.4),
            ("saturation", 0.0, 1.4),
            ("vibrance", -0.5, 0.5),
            ("black_lift", 0.0, 0.15),
            ("white_point", 0.85, 1.0),
            ("bloom", 0.0, 0.3),
            ("grain", 0.0, 0.025),
        ):
            _validate_number(name, getattr(self, name), low, high)


def _validate_number(name: str, value: float, low: float, high: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not low <= value <= high
    ):
        raise ValueError(f"{name} must be finite and in [{low}, {high}]")
