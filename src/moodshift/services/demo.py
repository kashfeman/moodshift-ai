"""Local CLI for photographic grades, neural reference transfer, and hybrid mode."""

import argparse
from pathlib import Path

from moodshift.imaging.exceptions import ImageError
from moodshift.imaging.io import export_image
from moodshift.presets import PRESETS
from moodshift.services.models import MoodSettings
from moodshift.services.mood_transfer import MoodTransferService


def main(argv: list[str] | None = None) -> int:
    """Report actual processing and export the result without overwriting files."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("content", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--preset", choices=[preset.id for preset in PRESETS])
    parser.add_argument("--reference")
    parser.add_argument(
        "--mode", choices=("grade", "neural", "hybrid"), default="grade"
    )
    parser.add_argument("--intensity", type=float, default=70)
    parser.add_argument(
        "--quality", choices=("preview", "balanced", "high"), default="balanced"
    )
    parser.add_argument("--max-size", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--content-preservation", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-grain", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)
    try:
        destination = Path(args.output)
        if destination.exists() or destination.is_symlink():
            raise ValueError("Output already exists; choose a new output path")
        if not destination.parent.is_dir():
            raise ValueError("Output parent directory must already exist")
        settings = MoodSettings(
            mode=args.mode,
            intensity=args.intensity,
            quality=args.quality,
            max_size=args.max_size,
            steps=args.steps,
            content_preservation=args.content_preservation,
            seed=args.seed,
            grain_enabled=not args.no_grain,
        )
        print(f"Starting {settings.mode} mode; intensity={settings.intensity:g}/100.")
        if settings.mode != "grade" and settings.intensity > 0:
            import torch

            torch.set_num_threads(
                2
            )  # This CLI process only; service changes no globals.
        service = MoodTransferService(
            device=args.device, allow_download=not args.offline
        )
        result = service.transfer(
            args.content,
            preset_id=args.preset,
            reference=args.reference,
            settings=settings,
        )
        export_image(result.image, args.output)
        facts = result.metadata
        print(
            f"Completed: engine={facts.actual_engine}, device={facts.device}, "
            f"reference={facts.reference_source}, "
            f"neural steps={facts.settings.neural_steps}, "
            f"output={result.image.width}x{result.image.height} RGB."
        )
    except (
        ImageError,
        ValueError,
        RuntimeError,
        ImportError,
        OSError,
        MemoryError,
    ) as error:
        parser.exit(1, f"MoodShift: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
