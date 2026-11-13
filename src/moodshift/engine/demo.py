"""Command-line demo of actual pretrained neural style transfer."""

import argparse
from dataclasses import replace

import torch

from moodshift.engine.exceptions import NeuralTransferError
from moodshift.engine.neural_transfer import NeuralStyleTransfer
from moodshift.engine.settings import TransferSettings
from moodshift.imaging.exceptions import ImageError
from moodshift.imaging.io import export_image


def main(argv: list[str] | None = None) -> int:
    """Run a local transfer and export safely, without logging input paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("content", "style", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--strength", type=float, default=0.5)
    parser.add_argument("--content-preservation", type=float, default=0.5)
    parser.add_argument("--quality", choices=("preview", "high"), default="preview")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--max-size", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--offline", action="store_true", help="Require cached official weights"
    )
    parser.add_argument(
        "--threads", type=int, default=2, help="CPU threads (1..16; default 2)"
    )
    args = parser.parse_args(argv)
    try:
        defaults = TransferSettings.for_quality(args.quality)
        settings = replace(
            defaults,
            style_strength=args.strength,
            content_preservation=args.content_preservation,
            seed=args.seed,
            steps=defaults.steps if args.steps is None else args.steps,
            max_size=defaults.max_size if args.max_size is None else args.max_size,
        )
        if not 1 <= args.threads <= 16:
            raise ValueError("threads must be in [1, 16]")
        torch.set_num_threads(args.threads)
        engine = NeuralStyleTransfer(
            device=args.device, allow_download=not args.offline
        )
        print(
            f"Starting neural transfer on {engine.device}; steps={settings.steps}, "
            f"max-size={settings.max_size}. First use may download official weights."
        )
        output = engine.transfer(args.content, args.style, settings)
        export_image(output, args.output)
        print(
            f"Completed on {engine.device}: "
            f"{output.width}x{output.height} RGB image exported."
        )
    except (ImageError, NeuralTransferError, ValueError) as error:
        parser.exit(1, f"MoodShift: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
