"""Bounded, immutable controls for optimization-based neural style transfer."""

import math
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class TransferSettings:
    """CPU-friendly preview defaults; output retains the processing resolution."""

    style_strength: float = 0.5
    content_preservation: float = 0.5
    steps: int = 60
    max_size: int = 192
    seed: int = 0

    def __post_init__(self) -> None:
        """Reject unsupported settings before image decoding or model loading."""
        for name in ("style_strength", "content_preservation"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError(f"{name} must be a finite number in [0, 1]")
        for name, lower, upper in (
            ("steps", 1, 500),
            ("max_size", 32, 512),
            ("seed", 0, 2**32 - 1),
        ):
            value = getattr(self, name)
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")

    @classmethod
    def for_quality(cls, quality: Literal["preview", "high"]) -> "TransferSettings":
        """Choose preview (60 steps/192px) or high (180 steps/384px) defaults."""
        if quality == "preview":
            return cls()
        if quality == "high":
            return cls(steps=180, max_size=384)
        raise ValueError("quality must be preview or high")
