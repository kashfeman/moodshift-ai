"""Immutable descriptive preset metadata; no image transformations."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MoodPreset:
    """Identify a planned visual style without defining processing behavior."""

    id: str
    name: str
    description: str

    def __post_init__(self) -> None:
        """Require meaningful metadata fields."""
        for field in ("id", "name", "description"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be a non-empty string")


PRESETS: tuple[MoodPreset, ...] = (
    MoodPreset("cinematic", "Cinematic", "Film-inspired contrast and dramatic tones."),
    MoodPreset("sunset", "Sunset", "Warm golden and orange tones inspired by dusk."),
    MoodPreset("cold-blue", "Cold Blue", "Cool blue tones for a crisp atmosphere."),
    MoodPreset(
        "vintage", "Vintage", "Muted colors and faded tones inspired by old photos."
    ),
    MoodPreset("dreamy", "Dreamy", "Soft contrast and gentle pastel tones."),
)


def get_preset(preset_id: str) -> MoodPreset:
    """Return metadata by stable ID, raising KeyError for an unknown ID."""
    for preset in PRESETS:
        if preset.id == preset_id:
            return preset
    raise KeyError(preset_id)
