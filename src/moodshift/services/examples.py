"""Generate reproducible local grading comparisons and an optional real hybrid demo."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from moodshift.imaging.io import export_image, load_image
from moodshift.imaging.reference_images import synthetic_scene
from moodshift.presets import PRESETS
from moodshift.services.models import MoodSettings
from moodshift.services.mood_transfer import MoodTransferService


def generate_examples(directory: Path, *, hybrid: bool = False) -> None:
    """Write synthetic inputs, real results, metadata, and a labeled contact sheet.

    A new directory is required. Hybrid uses cached official weights only; failed
    inference raises an error rather than being replaced by a grade-only result.
    """
    directory.mkdir(parents=True, exist_ok=False)
    original = synthetic_scene()
    source = export_image(original, directory / "original.png")
    service = MoodTransferService(device="cpu", allow_download=False)
    examples = [("Original (synthetic)", original)]
    for preset in PRESETS:
        result = service.transfer(
            source, preset_id=preset.id, settings=MoodSettings(intensity=100)
        )
        destination = export_image(result.image, directory / f"{preset.id}.png")
        reloaded = load_image(destination)
        difference = np.abs(
            np.asarray(reloaded).astype(float) - np.asarray(original)
        ).mean()
        if reloaded.size != original.size or difference <= 1:
            raise RuntimeError("A grading example failed pixel/size verification")
        print(f"{preset.name}: valid RGB, mean absolute change={difference:.3f}/255")
        examples.append((preset.name, result.image))
        with (directory / f"{preset.id}.json").open("x", encoding="utf-8") as stream:
            json.dump(asdict(result.metadata), stream, indent=2)
    sheet = Image.new(
        "RGB", (original.width * 3, (original.height + 24) * 2), "#171a23"
    )
    draw = ImageDraw.Draw(sheet)
    for index, (name, image) in enumerate(examples):
        left = (index % 3) * original.width
        top = (index // 3) * (original.height + 24)
        draw.text((left + 8, top + 6), name, fill="white")
        sheet.paste(image, (left, top + 24))
    export_image(sheet, directory / "comparison.png")
    if hybrid:
        import torch

        torch.set_num_threads(2)
        result = service.transfer(
            source,
            preset_id="cinematic",
            settings=MoodSettings(mode="hybrid", quality="preview", intensity=80),
        )
        destination = export_image(result.image, directory / "hybrid-cinematic.png")
        load_image(destination).close()
        with (directory / "hybrid-cinematic.json").open(
            "x", encoding="utf-8"
        ) as stream:
            json.dump(asdict(result.metadata), stream, indent=2)
        print(
            f"Hybrid completed: {result.metadata.actual_engine}, "
            f"device={result.metadata.device}, "
            f"reference={result.metadata.reference_source}, "
            f"steps={result.metadata.settings.neural_steps}"
        )


def main(argv: list[str] | None = None) -> int:
    """Generate examples in a new caller-selected directory without overwriting."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--hybrid", action="store_true", help="Use cached pretrained weights"
    )
    args = parser.parse_args(argv)
    try:
        generate_examples(args.output_dir, hybrid=args.hybrid)
    except (ValueError, RuntimeError, OSError, ImportError, MemoryError) as error:
        parser.exit(1, f"MoodShift examples: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
