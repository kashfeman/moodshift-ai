"""Compare trained guided-v1 and style-film-v1 on the same validation pairs."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch.nn import functional as F

from moodshift.engine.dataset import PairedImageDataset, inspect_dataset
from moodshift.engine.guided_decoder import GuidedDecoder, edge_loss
from moodshift.engine.model_loader import (
    load_feature_extractor,
    pretrained_checkpoint_path,
)
from moodshift.engine.style_conditioned_decoder import StyleConditionedDecoder
from moodshift.engine.tensors import tensor_to_image
from moodshift.engine.train_guided import dataset_fingerprint
from moodshift.engine.training import EncoderAdapter, align_features, feature_statistics


def compare(baseline: Path, conditioned: Path, data: Path, output: Path) -> None:
    """Evaluate four held-out pairings and create a new comparison directory."""
    checkpoints = [
        torch.load(p, map_location="cpu", weights_only=True)
        for p in (baseline, conditioned)
    ]
    if [c.get("architecture") for c in checkpoints] != ["guided-v1", "style-film-v1"]:
        raise ValueError("Expected guided-v1 baseline and style-film-v1 checkpoint")
    metas = [c.get("experiment", {}) for c in checkpoints]
    keys = (
        "seed",
        "epochs",
        "image_size",
        "batch_size",
        "optimizer",
        "learning_rate",
        "gradient_clip_norm",
        "device",
        "threads",
        "torch_version",
        "pairing",
        "dataset_sha256",
        "encoder_checkpoint_sha256",
        "loss_weights",
    )
    if any(k not in metas[0] or metas[0][k] != metas[1].get(k) for k in keys):
        raise ValueError(
            "Experiment settings differ; comparisons would not be controlled"
        )
    if metas[0].get("initial_decoder_sha256") != metas[1].get(
        "shared_baseline_initial_sha256"
    ):
        raise ValueError("Shared decoder layers had different initial weights")
    if any(c.get("epoch") != metas[0]["epochs"] for c in checkpoints):
        raise ValueError("Both checkpoints must have completed the configured epochs")
    report = inspect_dataset(data)
    if (
        not report.ready
        or dataset_fingerprint(data, report.files) != metas[0]["dataset_sha256"]
    ):
        raise ValueError("The current validated dataset does not match the experiments")
    with pretrained_checkpoint_path().open("rb") as stream:
        if (
            hashlib.file_digest(stream, "sha256").hexdigest()
            != metas[0]["encoder_checkpoint_sha256"]
        ):
            raise ValueError("The cached encoder does not match the experiments")
    torch.set_num_threads(2)
    encoder = EncoderAdapter(
        load_feature_extractor(torch.device("cpu"), allow_download=False)
    )
    old, new = GuidedDecoder(), StyleConditionedDecoder()
    old.load_state_dict(checkpoints[0]["decoder"], strict=True)
    new.load_state_dict(checkpoints[1]["decoder"], strict=True)
    old.eval()
    new.eval()
    dataset = PairedImageDataset(
        data, "val", size=metas[0]["image_size"], seed=metas[0]["seed"]
    )
    pairs = [dataset[i] for i in range(len(dataset))]
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    swaps = {"baseline": [], "conditioned": []}
    with torch.no_grad():
        for i, (content, _) in enumerate(pairs):
            content = content.unsqueeze(0)
            cf = encoder(content)
            outputs = {"baseline": [], "conditioned": []}
            for j, (_, style) in enumerate(pairs):
                style = style.unsqueeze(0)
                sf = encoder(style)
                target = align_features(cf, sf)
                generated = {
                    "baseline": old(target, content),
                    "conditioned": new(target, content, sf),
                }
                row = {"content": i + 1, "style": j + 1}
                for name, image in generated.items():
                    if not torch.isfinite(image).all():
                        raise ValueError("Nonfinite generated image")
                    outputs[name].append(image)
                    om, os = feature_statistics(encoder(image))
                    sm, ss = feature_statistics(sf)
                    row[f"{name}_rgb_mse"] = F.mse_loss(image, content).item()
                    row[f"{name}_edge_l1"] = edge_loss(image, content).item()
                    row[f"{name}_style_loss"] = (
                        (F.mse_loss(om, sm) + F.mse_loss(os, ss))
                        / (sm.square().mean() + ss.square().mean()).clamp_min(1e-5)
                    ).item()
                row["between_models_mae"] = (
                    (generated["baseline"] - generated["conditioned"])
                    .abs()
                    .mean()
                    .item()
                )
                rows.append(row)
                panel = Image.new("RGB", (384, 120), "white")
                draw = ImageDraw.Draw(panel)
                for column, (label, image) in enumerate(
                    (
                        ("Content", content),
                        ("Style", style),
                        ("Baseline", generated["baseline"]),
                        ("Conditioned", generated["conditioned"]),
                    )
                ):
                    panel.paste(tensor_to_image(image), (column * 96, 24))
                    draw.text((column * 96 + 2, 4), label, fill="black")
                with (output / f"content_{i + 1}_style_{j + 1}.png").open(
                    "xb"
                ) as stream:
                    panel.save(stream, format="PNG")
            for name in outputs:
                if len(outputs[name]) >= 2:
                    swaps[name].append(
                        (outputs[name][0] - outputs[name][1]).abs().mean().item()
                    )
    with (output / "metrics.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "pair_count": len(rows),
        "style_swap_mae_first_two_references": swaps,
        "note": "Mean absolute RGB difference in [0,1]; "
        "not a perceptual quality score.",
    }
    with (output / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)


def main() -> None:
    """Compare compatible checkpoints without training or overwriting results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--conditioned", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compare(args.baseline, args.conditioned, args.data, args.output)


if __name__ == "__main__":
    main()
