"""Run a small CPU-only content-guided decoder experiment."""

import argparse
import csv
import hashlib
import json
import math
from dataclasses import asdict
from functools import partial
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from moodshift.engine.color_conditioned_decoder import (
    ColorConditionedDecoder,
    color_objective,
    color_step,
)
from moodshift.engine.dataset import (
    inspect_dataset,
    make_data_loader,
    make_paired_reference_loader,
)
from moodshift.engine.guided_decoder import (
    BASELINE_WEIGHTS,
    STRONGER_STYLE_WEIGHTS,
    GuidedDecoder,
    GuidedLossWeights,
    guided_objective,
    guided_step,
)
from moodshift.engine.model_loader import (
    load_feature_extractor,
    pretrained_checkpoint_path,
)
from moodshift.engine.style_conditioned_decoder import (
    StyleConditionedDecoder,
    conditioned_objective,
    conditioned_step,
)
from moodshift.engine.tensors import tensor_to_image
from moodshift.engine.train_decoder import save_curve, validate_split
from moodshift.engine.training import EncoderAdapter, align_features


def dataset_fingerprint(data: Path, files: dict) -> str:
    """Hash sorted split/role paths and file contents to help compare runs."""
    digest = hashlib.sha256()
    for (split, role), paths in sorted(files.items()):
        for path in sorted(paths):
            relative = path.relative_to(data).as_posix().encode("utf-8")
            with path.open("rb") as stream:
                file_hash = hashlib.file_digest(stream, "sha256").digest()
            for value in (split.encode(), role.encode(), relative, file_hash):
                digest.update(len(value).to_bytes(8, "big"))
                digest.update(value)
    return digest.hexdigest()


def model_fingerprint(decoder: torch.nn.Module) -> str:
    """Hash the ordered initial decoder state without saving random weights."""
    digest = hashlib.sha256()
    for name, tensor in sorted(decoder.state_dict().items()):
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def shared_baseline_fingerprint(decoder: StyleConditionedDecoder) -> str:
    """Hash only the feature/refine layers shared with the baseline."""
    digest = hashlib.sha256()
    for name, tensor in sorted(decoder.state_dict().items()):
        if name.startswith(("feature.", "refine.")):
            digest.update(name.encode("utf-8") + b"\0")
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run(
    data: Path,
    output: Path,
    epochs: int = 5,
    seed: int = 0,
    weights: GuidedLossWeights = BASELINE_WEIGHTS,
    preset: str = "baseline",
    architecture: str = "guided-v1",
    reference_color_loss_weight: float = 0.0,
    paired_color_loss_weight: float = 0.0,
    paired_response_target: float = 0.25,
) -> None:
    """Train from scratch with frozen official weights and save each completed epoch."""
    if not 1 <= epochs <= 5:
        raise ValueError("epochs must be between 1 and 5")
    if not isinstance(weights, GuidedLossWeights):
        raise ValueError("weights must be a GuidedLossWeights instance")
    if architecture not in ("guided-v1", "style-film-v1", "color-film-v1"):
        raise ValueError("unsupported decoder architecture")
    if (
        isinstance(reference_color_loss_weight, bool)
        or not isinstance(reference_color_loss_weight, (int, float))
        or not math.isfinite(reference_color_loss_weight)
        or reference_color_loss_weight < 0
    ):
        raise ValueError("reference color loss weight must be finite and nonnegative")
    if reference_color_loss_weight and architecture != "color-film-v1":
        raise ValueError("reference color loss requires color-film-v1")
    for name, value in (
        ("paired color loss weight", paired_color_loss_weight),
        ("paired response target", paired_response_target),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"{name} must be finite and nonnegative")
    if paired_response_target > 1:
        raise ValueError("paired response target must be between 0 and 1")
    if paired_color_loss_weight and architecture != "color-film-v1":
        raise ValueError("paired color loss requires color-film-v1")
    report = inspect_dataset(data)
    validate_split(report)
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    encoder = EncoderAdapter(
        load_feature_extractor(torch.device("cpu"), allow_download=False)
    )
    decoder = {
        "guided-v1": GuidedDecoder,
        "style-film-v1": StyleConditionedDecoder,
        "color-film-v1": ColorConditionedDecoder,
    }[architecture]()
    step = {
        "guided-v1": guided_step,
        "style-film-v1": conditioned_step,
        "color-film-v1": color_step,
    }[architecture]
    objective = {
        "guided-v1": guided_objective,
        "style-film-v1": conditioned_objective,
        "color-film-v1": color_objective,
    }[architecture]
    if reference_color_loss_weight:
        step = partial(color_step, rgb_weight=reference_color_loss_weight)
        objective = partial(color_objective, rgb_weight=reference_color_loss_weight)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-4)
    with pretrained_checkpoint_path().open("rb") as stream:
        encoder_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    metadata = {
        "schema_version": 1,
        "architecture": architecture,
        "preset": preset,
        "loss_weights": asdict(weights),
        "seed": seed,
        "epochs": epochs,
        "image_size": 96,
        "batch_size": 1,
        "optimizer": "Adam",
        "learning_rate": 1e-4,
        "gradient_clip_norm": 1.0,
        "device": "cpu",
        "threads": 2,
        "torch_version": str(torch.__version__),
        "dataset_sha256": dataset_fingerprint(data, report.files),
        "initial_decoder_sha256": model_fingerprint(decoder),
        "encoder_checkpoint_sha256": encoder_hash,
        "pairing": "train seed + epoch; validation seed; sorted image paths",
    }
    if isinstance(decoder, StyleConditionedDecoder):
        metadata["shared_baseline_initial_sha256"] = shared_baseline_fingerprint(
            decoder
        )
    if isinstance(decoder, ColorConditionedDecoder):
        metadata["shared_style_film_initial_sha256"] = model_fingerprint(decoder.base)
        metadata["reference_color_loss_weight"] = reference_color_loss_weight
        metadata["paired_color_loss_weight"] = paired_color_loss_weight
        metadata["paired_response_target"] = paired_response_target
        if paired_color_loss_weight:
            metadata["second_reference_pairing"] = (
                "same split; seeded selection from byte-distinct sorted style files"
            )
    val = make_data_loader(data, "val", size=96, batch_size=1, seed=seed)
    paired_val = (
        make_paired_reference_loader(data, "val", size=96, batch_size=1, seed=seed)
        if paired_color_loss_weight
        else None
    )
    output.mkdir(parents=True, exist_ok=False)
    with (output / "experiment.json").open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
        stream.write("\n")
    history = []
    for epoch in range(1, epochs + 1):
        loader_factory = (
            make_paired_reference_loader
            if paired_color_loss_weight
            else make_data_loader
        )
        train = loader_factory(data, "train", size=96, batch_size=1, seed=seed + epoch)
        decoder.train()
        if paired_color_loss_weight:
            train_rows = [
                color_step(
                    encoder,
                    decoder,
                    optimizer,
                    c,
                    s,
                    weights,
                    rgb_weight=reference_color_loss_weight,
                    paired_style=s2,
                    paired_weight=paired_color_loss_weight,
                    response_target=paired_response_target,
                )
                for c, s, s2 in train
            ]
        else:
            train_rows = [
                step(encoder, decoder, optimizer, c, s, weights) for c, s in train
            ]
        decoder.eval()
        with torch.no_grad():
            if paired_val is not None:
                val_rows = [
                    {
                        k: v.item()
                        for k, v in color_objective(
                            encoder,
                            decoder,
                            c,
                            s,
                            weights,
                            rgb_weight=reference_color_loss_weight,
                            paired_style=s2,
                            paired_weight=paired_color_loss_weight,
                            response_target=paired_response_target,
                        ).items()
                    }
                    for c, s, s2 in paired_val
                ]
            else:
                val_rows = [
                    {
                        k: v.item()
                        for k, v in objective(encoder, decoder, c, s, weights).items()
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
        checkpoint = {
            "architecture": architecture,
            "epoch": epoch,
            "decoder": decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "history": history,
            "seed": seed,
            "experiment": metadata,
        }
        with (output / f"checkpoint_epoch_{epoch:03d}.pt").open("xb") as stream:
            torch.save(checkpoint, stream)
        with (output / f"losses_epoch_{epoch:03d}.csv").open(
            "x", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        print(
            f"epoch {epoch}: train={row['train_loss']:.6f} val={row['val_loss']:.6f}",
            flush=True,
        )
    # Final convenience files are created once; epoch snapshots are never replaced.
    for source, target in (
        (output / f"checkpoint_epoch_{epochs:03d}.pt", output / "checkpoint.pt"),
        (output / f"losses_epoch_{epochs:03d}.csv", output / "losses.csv"),
    ):
        with source.open("rb") as input_stream, target.open("xb") as output_stream:
            output_stream.write(input_stream.read())
    with (output / "loss_curve.png").open("xb") as stream:
        save_curve(history, stream)
    saved = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True)
    model = {
        "guided-v1": GuidedDecoder,
        "style-film-v1": StyleConditionedDecoder,
        "color-film-v1": ColorConditionedDecoder,
    }[architecture]()
    model.load_state_dict(saved["decoder"])
    model.eval()
    with torch.no_grad():
        for index, (c, s) in enumerate(val.dataset, 1):
            c, s = c.unsqueeze(0), s.unsqueeze(0)
            style_features = encoder(s)
            target = align_features(encoder(c), style_features)
            if architecture == "guided-v1":
                generated = model(target, c)
            elif architecture == "style-film-v1":
                generated = model(target, c, style_features)
            else:
                generated = model(target, c, style_features, s)
            for label, pixels in (
                ("content", c),
                ("style", s),
                ("guided", generated),
            ):
                with (output / f"val_{index}_{label}.png").open("xb") as stream:
                    tensor_to_image(pixels).save(stream, format="PNG")
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
        with (output / f"val_{index}_comparison.png").open("xb") as stream:
            canvas.save(stream, format="PNG")


def main() -> None:
    """Train a fresh experiment and optionally compare an earlier run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--architecture",
        choices=("guided-v1", "style-film-v1", "color-film-v1"),
        default="guided-v1",
    )
    parser.add_argument(
        "--loss-preset", choices=("baseline", "stronger-style"), default="baseline"
    )
    for name in ("content", "style", "identity", "edge"):
        parser.add_argument(f"--{name}-weight", type=float, default=None)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--reference-color-loss-weight", type=float, default=0.0)
    parser.add_argument("--paired-color-loss-weight", type=float, default=0.0)
    parser.add_argument("--paired-response-target", type=float, default=0.25)
    args = parser.parse_args()
    defaults = (
        BASELINE_WEIGHTS if args.loss_preset == "baseline" else STRONGER_STYLE_WEIGHTS
    )
    values = {
        name: getattr(args, f"{name}_weight")
        if getattr(args, f"{name}_weight") is not None
        else getattr(defaults, name)
        for name in ("content", "style", "identity", "edge")
    }
    try:
        weights = GuidedLossWeights(**values)
    except ValueError as error:
        parser.error(str(error))
    if args.paired_color_loss_weight or args.paired_response_target != 0.25:
        run(
            args.data,
            args.output,
            args.epochs,
            args.seed,
            weights,
            args.loss_preset,
            args.architecture,
            args.reference_color_loss_weight,
            args.paired_color_loss_weight,
            args.paired_response_target,
        )
    elif args.architecture == "guided-v1" and args.reference_color_loss_weight == 0:
        run(args.data, args.output, args.epochs, args.seed, weights, args.loss_preset)
    else:
        run(
            args.data,
            args.output,
            args.epochs,
            args.seed,
            weights,
            args.loss_preset,
            args.architecture,
            args.reference_color_loss_weight,
        )
    if args.previous:
        compare(args.output, args.previous)


if __name__ == "__main__":
    main()
