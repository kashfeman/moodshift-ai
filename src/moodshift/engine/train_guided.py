"""Run a small CPU-only content-guided decoder experiment."""

import argparse
import csv
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from moodshift.engine.dataset import inspect_dataset, make_data_loader
from moodshift.engine.guided_decoder import GuidedDecoder, guided_objective, guided_step
from moodshift.engine.model_loader import load_feature_extractor
from moodshift.engine.tensors import tensor_to_image
from moodshift.engine.train_decoder import save_curve, validate_split
from moodshift.engine.training import EncoderAdapter, align_features


def run(data: Path, output: Path, epochs: int = 5, seed: int = 0) -> None:
    """Train from scratch with frozen official weights and save each completed epoch."""
    if not 1 <= epochs <= 5:
        raise ValueError("epochs must be between 1 and 5")
    validate_split(inspect_dataset(data))
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    encoder = EncoderAdapter(
        load_feature_extractor(torch.device("cpu"), allow_download=False)
    )
    decoder = GuidedDecoder()
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-4)
    val = make_data_loader(data, "val", size=96, batch_size=1, seed=seed)
    output.mkdir(parents=True, exist_ok=False)
    history = []
    for epoch in range(1, epochs + 1):
        train = make_data_loader(
            data, "train", size=96, batch_size=1, seed=seed + epoch
        )
        decoder.train()
        train_rows = [guided_step(encoder, decoder, optimizer, c, s) for c, s in train]
        decoder.eval()
        with torch.no_grad():
            val_rows = [
                {
                    k: v.item()
                    for k, v in guided_objective(encoder, decoder, c, s).items()
                }
                for c, s in val
            ]
        if not all(
            torch.isfinite(torch.tensor(value))
            for row in val_rows
            for value in row.values()
        ):
            raise ValueError("nonfinite validation loss")
        row = {"epoch": epoch}
        for label, rows in (("train", train_rows), ("val", val_rows)):
            row.update(
                {
                    f"{label}_{key}": sum(r[key] for r in rows) / len(rows)
                    for key in rows[0]
                }
            )
        history.append(row)
        torch.save(
            {
                "architecture": "guided-v1",
                "epoch": epoch,
                "decoder": decoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "history": history,
                "seed": seed,
            },
            output / "checkpoint.pt",
        )
        with (output / "losses.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        save_curve(history, output / "loss_curve.png")
        print(
            f"epoch {epoch}: train={row['train_loss']:.6f} val={row['val_loss']:.6f}",
            flush=True,
        )
    saved = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True)
    model = GuidedDecoder()
    model.load_state_dict(saved["decoder"])
    model.eval()
    with torch.no_grad():
        for index, (c, s) in enumerate(val.dataset, 1):
            c, s = c.unsqueeze(0), s.unsqueeze(0)
            target = align_features(encoder(c), encoder(s))
            tensor_to_image(c).save(output / f"val_{index}_content.png")
            tensor_to_image(s).save(output / f"val_{index}_style.png")
            tensor_to_image(model(target, c)).save(output / f"val_{index}_guided.png")
    print("Reloaded checkpoint and generated validation images", flush=True)


def compare(output: Path, previous: Path) -> None:
    """Save a truthful four-image comparison after verifying exact prior inputs."""
    for index in range(1, len(list(output.glob("val_*_content.png"))) + 1):
        content = Image.open(output / f"val_{index}_content.png").convert("RGB")
        style = Image.open(output / f"val_{index}_style.png").convert("RGB")
        if (
            content.tobytes()
            != Image.open(previous / f"val_{index}_before.png").convert("RGB").tobytes()
        ):
            raise ValueError("Previous validation content does not match")
        if (
            style.tobytes()
            != Image.open(previous / f"val_{index}_style.png").convert("RGB").tobytes()
        ):
            raise ValueError("Previous validation style does not match")
        old = Image.open(previous / f"val_{index}_after.png").convert("RGB")
        new = Image.open(output / f"val_{index}_guided.png").convert("RGB")
        canvas = Image.new("RGB", (384, 120), "white")
        draw = ImageDraw.Draw(canvas)
        for col, (label, image) in enumerate(
            zip(
                ("Content", "Style", "7C.1", "7C.2"),
                (content, style, old, new),
                strict=True,
            )
        ):
            canvas.paste(image, (col * 96, 24))
            draw.text((col * 96 + 2, 4), label, fill="black")
        canvas.save(output / f"val_{index}_comparison.png")


def main() -> None:
    """Train a fresh experiment and optionally compare an earlier run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--previous", type=Path)
    args = parser.parse_args()
    run(args.data, args.output, args.epochs, args.seed)
    if args.previous:
        compare(args.output, args.previous)


if __name__ == "__main__":
    main()
