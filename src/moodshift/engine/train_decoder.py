"""CPU decoder training experiment with validation and resumable checkpoints."""

import argparse
import csv
import math
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from moodshift.engine.dataset import inspect_dataset, make_data_loader
from moodshift.engine.model_loader import load_feature_extractor
from moodshift.engine.tensors import tensor_to_image
from moodshift.engine.training import (
    EncoderAdapter,
    ImageDecoder,
    TrainingConfig,
    align_features,
    decoder_objective,
    make_optimizer,
    training_step,
)


def evaluate_batch(encoder, decoder, content, style, config):
    """Compute the same objective as training without updating any parameters."""
    with torch.no_grad():
        losses = decoder_objective(encoder, decoder, content, style, config)
        if not all(torch.isfinite(value) for value in losses.values()):
            raise ValueError("validation loss is not finite")
        return {name: value.item() for name, value in losses.items()}


def validate_split(report):
    """Reject invalid data, insufficient counts, and identical cross-split pixels."""
    import hashlib

    from moodshift.imaging.io import load_image

    if not report.ready:
        raise ValueError(
            f"Dataset validation failed: {report.problems}; "
            f"duplicates: {report.duplicates}"
        )
    seen = {}
    for (split, _role), paths in report.files.items():
        for path in paths:
            image = load_image(path)
            digest = (image.size, hashlib.sha256(image.tobytes()).digest())
            if digest in seen and seen[digest][0] != split:
                raise ValueError(
                    f"Identical decoded pixels across splits: "
                    f"{seen[digest][1]} and {path}"
                )
            seen[digest] = (split, path)


def average_batches(rows):
    """Average losses weighted by batch size."""
    count = sum(size for size, _ in rows)
    if not count:
        raise ValueError("Dataset produced no batches")
    return {
        key: sum(size * row[key] for size, row in rows) / count
        for key in ("loss", "content_loss", "style_loss", "reconstruction_loss")
    }


def save_curve(history, path):
    """Draw actual per-epoch total losses with Pillow, without extra dependencies."""
    image = Image.new("RGB", (800, 500), "white")
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = 80, 50, 760, 420
    draw.line((left, top, left, bottom, right, bottom), fill="black", width=2)
    values = [row[key] for row in history for key in ("train_loss", "val_loss")]
    low, high = min(values), max(values)
    if high == low:
        high = low + 1
    draw.text(
        (left, 15), "Decoder total loss per epoch (train / validation)", fill="black"
    )
    draw.text((5, top), f"{high:.4g}", fill="black")
    draw.text((5, bottom - 10), f"{low:.4g}", fill="black")
    for key, color in (("train_loss", "blue"), ("val_loss", "red")):
        points = []
        for index, row in enumerate(history):
            x = left + (right - left) * index / max(1, len(history) - 1)
            y = bottom - (bottom - top) * (row[key] - low) / (high - low)
            points.append((x, y))
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
            if key == "train_loss":
                draw.text((x - 5, bottom + 8), str(row["epoch"]), fill="black")
        if len(points) > 1:
            draw.line(points, fill=color, width=2)
    draw.text((left, 460), "Epoch     blue: train     red: validation", fill="black")
    image.save(path)


def export_results(history, directory):
    """Export recorded epoch metrics and a plot derived from those metrics."""
    with (directory / "losses.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    save_curve(history, directory / "loss_curve.png")


def run(args):
    """Run a new or resumed CPU experiment and save completed epochs only."""
    if args.epochs < 1 or args.threads < 1:
        raise ValueError("epochs and threads must be positive")
    config = TrainingConfig(learning_rate=args.learning_rate)
    report = inspect_dataset(args.data)
    validate_split(report)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    encoder = EncoderAdapter(
        load_feature_extractor(torch.device("cpu"), allow_download=False)
    )
    decoder = ImageDecoder()
    optimizer = make_optimizer(decoder, config)
    history = []
    start = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        if checkpoint.get("objective_version") != 2:
            raise ValueError("Checkpoint uses an incompatible training objective")
        expected = {
            "size": args.size,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "learning_rate": args.learning_rate,
        }
        if checkpoint["settings"] != expected:
            raise ValueError("Resume settings must match checkpoint settings")
        decoder.load_state_dict(checkpoint["decoder"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        history = checkpoint["history"]
        start = checkpoint["epoch"]
        if args.epochs <= start:
            raise ValueError("epochs must exceed the checkpoint epoch")
    directory = args.output
    directory.mkdir(parents=True, exist_ok=False)
    settings = {
        "size": args.size,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
    }
    val = make_data_loader(
        args.data, "val", size=args.size, batch_size=args.batch_size, seed=args.seed
    )
    for epoch in range(start + 1, args.epochs + 1):
        train = make_data_loader(
            args.data,
            "train",
            size=args.size,
            batch_size=args.batch_size,
            seed=args.seed + epoch,
        )
        decoder.train()
        train_rows = [
            (
                content.shape[0],
                training_step(encoder, decoder, optimizer, content, style, config),
            )
            for content, style in train
        ]
        decoder.eval()
        val_rows = [
            (content.shape[0], evaluate_batch(encoder, decoder, content, style, config))
            for content, style in val
        ]
        train_avg, val_avg = average_batches(train_rows), average_batches(val_rows)
        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_avg.items()},
            **{f"val_{k}": v for k, v in val_avg.items()},
        }
        if not all(math.isfinite(float(value)) for value in row.values()):
            raise ValueError("Nonfinite epoch metrics")
        history.append(row)
        torch.save(
            {
                "epoch": epoch,
                "objective_version": 2,
                "decoder": decoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "history": history,
                "settings": settings,
            },
            directory / "checkpoint.pt",
        )
        export_results(history, directory)
        print(
            f"epoch {epoch}: train={train_avg['loss']:.6f} val={val_avg['loss']:.6f}",
            flush=True,
        )
    # Reload the saved decoder for actual held-out inference.
    saved = torch.load(
        directory / "checkpoint.pt", map_location="cpu", weights_only=True
    )
    reloaded = ImageDecoder()
    reloaded.load_state_dict(saved["decoder"])
    reloaded.eval()
    with torch.no_grad():
        for index, (content, style) in enumerate(val.dataset):
            target = align_features(
                encoder(content.unsqueeze(0)), encoder(style.unsqueeze(0))
            )
            output = reloaded(target, tuple(content.shape[-2:]))
            tensor_to_image(content.unsqueeze(0)).save(
                directory / f"val_{index + 1}_before.png"
            )
            tensor_to_image(style.unsqueeze(0)).save(
                directory / f"val_{index + 1}_style.png"
            )
            tensor_to_image(output).save(directory / f"val_{index + 1}_after.png")
    print("Reloaded checkpoint and generated validation images", flush=True)


def main():
    """Parse the CLI and execute a CPU-only training run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--epochs", type=int, default=5, help="total epochs including resumed epochs"
    )
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--size", type=int, choices=(96, 128), default=96)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--resume", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
