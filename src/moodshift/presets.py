"""Single immutable registry of photographic mood metadata and processing controls."""

from dataclasses import dataclass

from moodshift.imaging.grading_parameters import GradeParameters


@dataclass(frozen=True, slots=True)
class MoodPreset:
    """Identify a visual style and its immutable photographic processing controls."""

    id: str
    name: str
    description: str
    grading: GradeParameters = GradeParameters()

    def __post_init__(self) -> None:
        """Require meaningful metadata fields."""
        for field in ("id", "name", "description"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be a non-empty string")
        if not isinstance(self.grading, GradeParameters):
            raise ValueError("grading must contain GradeParameters")


PRESETS: tuple[MoodPreset, ...] = (
    MoodPreset(
        "cinematic",
        "Cinematic",
        "Film-inspired contrast and dramatic tones.",
        GradeParameters(
            shadows=(-5, -7),
            midtones=(1, 1),
            highlights=(3, 7),
            contrast=1.16,
            saturation=0.94,
            vibrance=0.12,
            black_lift=0.008,
            white_point=0.985,
        ),
    ),
    MoodPreset(
        "sunset",
        "Sunset",
        "Warm golden and orange tones inspired by dusk.",
        GradeParameters(
            shadows=(3, 1),
            midtones=(5, 8),
            highlights=(2, 12),
            gamma=1.05,
            saturation=1.04,
            vibrance=0.12,
            contrast=1.02,
            white_point=0.99,
        ),
    ),
    MoodPreset(
        "cold-blue",
        "Cold Blue",
        "Cool blue tones for a crisp atmosphere.",
        GradeParameters(
            shadows=(-4, -10),
            midtones=(-2, -7),
            highlights=(-1, -4),
            gamma=0.98,
            contrast=1.07,
            saturation=0.86,
            black_lift=0.015,
            white_point=0.98,
        ),
    ),
    MoodPreset(
        "vintage",
        "Vintage",
        "Muted colors and faded tones inspired by old photos.",
        GradeParameters(
            shadows=(2, 3),
            midtones=(3, 5),
            highlights=(1, 7),
            contrast=0.90,
            gamma=1.03,
            saturation=0.77,
            black_lift=0.055,
            white_point=0.95,
            grain=0.006,
        ),
    ),
    MoodPreset(
        "dreamy",
        "Dreamy",
        "Soft contrast and gentle pastel tones.",
        GradeParameters(
            shadows=(1, -2),
            midtones=(2, 0),
            highlights=(3, 1),
            contrast=0.83,
            gamma=1.08,
            saturation=0.82,
            black_lift=0.025,
            white_point=0.985,
            bloom=0.18,
        ),
    ),
)


def get_preset(preset_id: str) -> MoodPreset:
    """Return metadata by stable ID, raising KeyError for an unknown ID."""
    for preset in PRESETS:
        if preset.id == preset_id:
            return preset
    raise KeyError(preset_id)
