"""Temporary synthetic references bridge to the unchanged path-based neural API."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from moodshift.imaging.io import export_image
from moodshift.imaging.reference_images import synthetic_style_reference
from moodshift.presets import MoodPreset, get_preset
from moodshift.services.models import Mode


def resolve_preset(
    preset_id: str | None, reference: str | Path | None, mode: Mode
) -> MoodPreset | None:
    """Validate mode/reference selection without silently ignoring user controls."""
    try:
        preset = get_preset(preset_id) if preset_id is not None else None
    except KeyError:
        raise ValueError("Unknown preset ID") from None
    if mode in ("grade", "hybrid") and preset is None:
        raise ValueError("grade and hybrid modes require a preset")
    if mode == "grade" and reference is not None:
        raise ValueError("grade mode does not use a reference; select neural or hybrid")
    if mode == "neural" and reference is None and preset is None:
        raise ValueError("neural mode needs a reference or a preset")
    if mode == "neural" and reference is not None and preset is not None:
        raise ValueError("Use hybrid to combine a named preset with a user reference")
    return preset


@contextmanager
def reference_path(reference: Path | None, preset: MoodPreset | None) -> Iterator[Path]:
    """Use a user path or create/clean up one deterministic synthetic reference PNG."""
    if reference is not None:
        yield reference
        return
    if preset is None:
        raise ValueError("Neural transfer needs a reference or a preset")
    with TemporaryDirectory(prefix="moodshift-reference-") as directory:
        path = Path(directory) / "synthetic.png"
        export_image(synthetic_style_reference(preset.grading), path)
        yield path
