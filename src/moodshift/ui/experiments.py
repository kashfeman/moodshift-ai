"""Read a pinned local experiment and run its trained decoder on CPU."""

import csv
import hashlib
import io
import math
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageOps

from moodshift.imaging.io import export_image, load_image

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUN_NAME = "experiment_7c2"
CHECKPOINT_SHA256 = "59594434fb728daf2f9dc5cc27ec0e911ce4755819fcb05f54c0cf83b4c5a485"


class ExperimentError(ValueError):
    """A trained experiment is missing, invalid, or cannot run safely."""


def load_checkpoint(root: Path = PROJECT_ROOT) -> dict:
    """Load only the exact project experiment, validating its hash and contents."""
    import torch

    from moodshift.engine.guided_decoder import GuidedDecoder

    path = root / "checkpoints" / RUN_NAME / "checkpoint.pt"
    try:
        if path.stat().st_size > 30_000_000:
            raise ExperimentError("The experiment checkpoint is too large.")
        data = path.read_bytes()
    except OSError as error:
        raise ExperimentError(
            "The trained checkpoint is not available locally."
        ) from error
    if hashlib.sha256(data).hexdigest() != CHECKPOINT_SHA256:
        raise ExperimentError(
            "The checkpoint does not match the trusted project experiment."
        )
    try:
        checkpoint = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        if (
            not isinstance(checkpoint, dict)
            or checkpoint.get("architecture") != "guided-v1"
        ):
            raise ExperimentError("The checkpoint architecture is incompatible.")
        history = checkpoint["history"]
        epoch = checkpoint["epoch"]
        if (
            type(epoch) is not int
            or epoch < 1
            or not isinstance(history, list)
            or len(history) != epoch
        ):
            raise ExperimentError("The checkpoint training history is invalid.")
        for index, row in enumerate(history, 1):
            if row["epoch"] != index or not all(
                math.isfinite(float(row[k])) for k in ("train_loss", "val_loss")
            ):
                raise ExperimentError("The checkpoint training history is invalid.")
        with torch.random.fork_rng(devices=[]):
            model = GuidedDecoder()
        model.load_state_dict(checkpoint["decoder"], strict=True)
        if not all(
            torch.isfinite(value).all() for value in model.state_dict().values()
        ):
            raise ExperimentError("The checkpoint contains nonfinite weights.")
        if not checkpoint.get("optimizer", {}).get("state"):
            raise ExperimentError("The checkpoint has no recorded optimizer updates.")
        return checkpoint
    except (KeyError, TypeError, ValueError, RuntimeError, EOFError, OSError) as error:
        if isinstance(error, ExperimentError):
            raise
        raise ExperimentError(
            "The trained checkpoint is incompatible or unreadable."
        ) from error


def experiment_info(
    root: Path = PROJECT_ROOT,
) -> tuple[str, list[list], Image.Image | None, list[tuple[Image.Image, str]], bool]:
    """Read evidence without exposing filesystem paths or inventing missing values."""
    import torch

    from moodshift.engine.dataset import inspect_dataset
    from moodshift.engine.guided_decoder import GuidedDecoder
    from moodshift.engine.model_loader import load_feature_extractor

    messages = []
    rows: list[list] = []
    available = False
    try:
        checkpoint = load_checkpoint(root)
        with torch.random.fork_rng(devices=[]):
            count = sum(
                p.numel() for p in GuidedDecoder().parameters() if p.requires_grad
            )
        history = checkpoint["history"]
        rows = [[r["epoch"], r["train_loss"], r["val_loss"]] for r in history]
        messages.append(
            f"**Verified checkpoint:** guided-v1, {count:,} trainable parameters; "
            f"{checkpoint['epoch']} epochs; seed {checkpoint['seed']}. "
            f"Final training loss {history[-1]['train_loss']:.6f}; "
            f"validation loss {history[-1]['val_loss']:.6f}."
        )
        try:
            load_feature_extractor(torch.device("cpu"), allow_download=False)
            available = True
        except (RuntimeError, ValueError, OSError):
            messages.append(
                "Official SqueezeNet weights could not be loaded from the local "
                "cache; inference is unavailable."
            )
    except ExperimentError as error:
        messages.append(str(error))
    messages.append(
        "**Encoder:** official ImageNet SqueezeNet 1.1, frozen during training. "
        "The experiment script specifies CPU, Adam 0.0001, 96 px, batch size 1, "
        "gradient clipping and separate validation. Those settings and historical "
        "dataset counts were not embedded in this checkpoint."
    )
    try:
        report = inspect_dataset(root / "data")
        counts = [
            len(report.files[split, role])
            for split in ("train", "val")
            for role in ("content", "style")
        ]
        messages.append(
            "**Current dataset inventory (not proof of historical training counts):** "
            f"train content/style {counts[0]}/{counts[1]}; "
            f"validation content/style {counts[2]}/{counts[3]}."
        )
    except OSError:
        messages.append("Current dataset inventory is unavailable.")
    directory = root / "checkpoints" / RUN_NAME
    graph = None
    try:
        with Image.open(directory / "loss_curve.png") as image:
            image.load()
            graph = image.convert("RGB")
    except (OSError, ValueError):
        messages.append(
            "Loss-curve image is unavailable; checkpoint metrics are listed "
            "when available."
        )
    samples = []
    for index in (1, 2):
        for key, label in (
            ("content", "Content"),
            ("style", "Style reference"),
            ("guided", "Generated"),
        ):
            try:
                with Image.open(directory / f"val_{index}_{key}.png") as image:
                    image.load()
                    samples.append(
                        (image.convert("RGB"), f"Validation {index}: {label}")
                    )
            except (OSError, ValueError):
                messages.append(
                    f"Validation {index} {label.lower()} image is unavailable."
                )
    # Compare the optional CSV against the checkpoint.
    try:
        with (directory / "losses.csv").open(
            newline="", encoding="utf-8-sig"
        ) as stream:
            csv_rows = list(csv.DictReader(stream))
        if rows and (
            len(csv_rows) != len(rows)
            or any(
                int(r["epoch"]) != int(expected[0])
                or not math.isclose(float(r["train_loss"]), expected[1], rel_tol=1e-7)
                or not math.isclose(float(r["val_loss"]), expected[2], rel_tol=1e-7)
                for r, expected in zip(csv_rows, rows, strict=True)
            )
        ):
            messages.append(
                "CSV metrics do not match the checkpoint; showing checkpoint history."
            )
    except (OSError, ValueError, KeyError):
        messages.append(
            "The metrics CSV is unavailable or invalid; checkpoint history is "
            "shown when available."
        )
    return "\n\n".join(messages), rows, graph, samples, available


def infer(
    content_path: str | None,
    style_path: str | None,
    selection: str | None,
    output_dir: Path,
    root: Path = PROJECT_ROOT,
) -> tuple[Image.Image, tuple[Image.Image, Image.Image], str]:
    """Run verified guided weights on validated 96px uploads and export a PNG."""
    import torch

    from moodshift.engine.guided_decoder import GuidedDecoder
    from moodshift.engine.model_loader import load_feature_extractor
    from moodshift.engine.tensors import image_to_tensor, tensor_to_image
    from moodshift.engine.training import EncoderAdapter, align_features

    if not content_path or not style_path:
        raise ExperimentError("Upload both a content image and a style reference.")
    if selection != RUN_NAME:
        raise ExperimentError("Choose the available trusted checkpoint.")
    checkpoint = load_checkpoint(root)
    content = ImageOps.fit(
        load_image(content_path), (96, 96), method=Image.Resampling.LANCZOS
    )
    style = ImageOps.fit(
        load_image(style_path), (96, 96), method=Image.Resampling.LANCZOS
    )
    encoder = EncoderAdapter(
        load_feature_extractor(torch.device("cpu"), allow_download=False)
    )
    with torch.random.fork_rng(devices=[]):
        decoder = GuidedDecoder()
    decoder.load_state_dict(checkpoint["decoder"], strict=True)
    decoder.eval()
    with torch.inference_mode():
        pixels = image_to_tensor(content, torch.device("cpu"))
        reference = image_to_tensor(style, torch.device("cpu"))
        target = align_features(encoder(pixels), encoder(reference))
        result = tensor_to_image(decoder(target, pixels))
    download = export_image(result, output_dir / f"moodshift-{uuid4().hex}.png")
    return result, (content, result), str(download)
