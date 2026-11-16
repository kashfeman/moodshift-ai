"""Pure mode handling, friendly failures, and truthful metadata presentation."""

from dataclasses import asdict
from typing import Any

from moodshift.imaging.exceptions import ImageError
from moodshift.services.models import MoodMetadata
from moodshift.ui.state import ModePresentation


def mode_presentation(
    mode: str, has_reference: bool, intensity: float, style_strength: float = 1.0
) -> ModePresentation:
    """Explain which stages/reference will run, including the zero-style bypass."""
    if mode == "grade":
        explanation = (
            "**Photographic grade** · CIELAB color and tone processing. "
            "No neural model or reference is used."
        )
    elif mode == "neural":
        source = (
            "your uploaded reference"
            if has_reference
            else "the selected preset's synthetic built-in reference"
        )
        explanation = (
            "**Real neural transfer** · Frozen pretrained SqueezeNet features "
            f"guide optimization using {source}. No post-grading."
        )
    elif mode == "hybrid":
        source = (
            "your uploaded reference"
            if has_reference
            else "a synthetic built-in preset reference"
        )
        explanation = (
            f"**Hybrid transfer** · Real SqueezeNet optimization using {source}, "
            "with content-guided reconstruction, then the selected photographic grade."
        )
    else:
        raise ValueError("Choose grade, neural, or hybrid mode.")
    if intensity == 0:
        explanation += (
            "\n\n**Zero style intensity:** exact normalized content pixels; "
            "no model, grade, resize, or reference application."
        )
    elif mode != "grade" and style_strength == 0:
        explanation += (
            "\n\n**Neural style strength is zero:** no model or reference is applied. "
            + (
                "Only the selected photographic grade runs."
                if mode == "hybrid"
                else "The normalized content is returned unchanged."
            )
        )
    return ModePresentation(
        mode != "grade",
        not (mode == "neural" and has_reference),
        mode in ("grade", "hybrid"),
        explanation,
    )


def format_metadata(metadata: MoodMetadata) -> tuple[str, dict[str, Any]]:
    """Format actual device, inference, reference, and effective settings facts."""
    settings = metadata.settings
    strength = (
        f"{settings.neural_style_strength:g}"
        if settings.neural_style_strength is not None
        else "Not used"
    )
    reference = {
        "none": "None applied",
        "user": "Uploaded reference",
        "synthetic-preset": "Synthetic built-in preset reference",
    }[metadata.reference_source]
    summary = (
        f"**Mode:** {metadata.mode} · "
        f"**Preset:** {metadata.preset_id or 'User reference'}  \n"
        f"**Engine:** {metadata.actual_engine} · "
        f"**Device:** {metadata.device or 'Not used'}  \n"
        f"**Reference:** {reference}  \n"
        f"**Output:** {metadata.output_size[0]} × {metadata.output_size[1]} RGB · "
        f"**Intensity:** {settings.intensity:g}/100 · "
        f"**Neural steps:** {settings.neural_steps} · "
        f"**Neural strength:** {strength}  \n"
        f"**Reconstruction:** {metadata.reconstruction}"
    )
    return summary, asdict(metadata)


def friendly_error(error: Exception) -> str:
    """Keep user-facing failures useful while withholding arbitrary filesystem paths."""
    if isinstance(error, (ImageError, ValueError)):
        return str(error)
    if isinstance(error, MemoryError):
        return (
            "Not enough memory. Enable an output size limit and choose Preview quality."
        )
    if isinstance(error, (RuntimeError, ImportError)):
        return (
            "Processing could not complete. Check the selected neural device "
            "and cached SqueezeNet weights. Use CPU/Preview, or turn Offline off to "
            "allow the official first-run download. No substitute result was returned."
        )
    if isinstance(error, OSError):
        return "The local image/download could not be accessed. Re-upload and retry."
    return "An unexpected error stopped this request. No result was returned; retry."
