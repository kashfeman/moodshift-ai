"""Regression tests for CPU training, checkpoint resume, and data isolation."""

from argparse import Namespace
from pathlib import Path

import pytest
import torch
from PIL import Image
from torchvision.models import squeezenet1_1

from moodshift.engine.dataset import inspect_dataset
from moodshift.engine.features import SqueezeNetFeatures
from moodshift.engine.train_decoder import run, validate_split


def make_dataset(root: Path) -> None:
    """Write synthetic, distinct fixtures into the four required directories."""
    for split_index, split in enumerate(("train", "val")):
        for role_index, role in enumerate(("content", "style")):
            folder = root / split / role
            folder.mkdir(parents=True)
            for i in range(10 if split == "train" else 2):
                Image.new(
                    "RGB", (32, 35), (i * 10, role_index * 90, split_index * 100)
                ).save(folder / f"{i}.png")


def test_rejects_decoded_duplicate(tmp_path: Path) -> None:
    """Catch exact pixel copies even if PNG encoding bytes differ."""
    make_dataset(tmp_path)
    source = tmp_path / "train/content/0.png"
    target = tmp_path / "val/style/copy.png"
    with Image.open(source) as image:
        image.save(target, compress_level=0)
    report = inspect_dataset(tmp_path)
    assert not report.duplicates
    with pytest.raises(ValueError, match="Identical decoded pixels"):
        validate_split(report)


def test_train_resume_and_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resume optimizer state and reproduce an uninterrupted two-epoch run."""
    import moodshift.engine.train_decoder as module

    make_dataset(tmp_path / "data")
    monkeypatch.setattr(
        module,
        "load_feature_extractor",
        lambda device, allow_download: SqueezeNetFeatures(
            squeezenet1_1(weights=None).features
        ),
    )

    def args(output: str, epochs: int, resume: Path | None = None) -> Namespace:
        return Namespace(
            data=tmp_path / "data",
            output=tmp_path / output,
            epochs=epochs,
            learning_rate=0.001,
            batch_size=1,
            size=96,
            seed=3,
            threads=2,
            resume=resume,
        )

    previous = torch.get_num_threads()
    try:
        run(args("first", 1))
        checkpoint = tmp_path / "first/checkpoint.pt"
        run(args("resumed", 2, checkpoint))
        run(args("whole", 2))
    finally:
        torch.set_num_threads(previous)
    resumed = torch.load(tmp_path / "resumed/checkpoint.pt", weights_only=True)
    whole = torch.load(tmp_path / "whole/checkpoint.pt", weights_only=True)
    assert resumed["epoch"] == 2
    assert resumed["history"] == whole["history"]
    assert all(
        torch.equal(resumed["decoder"][key], value)
        for key, value in whole["decoder"].items()
    )
    assert (tmp_path / "resumed/losses.csv").is_file()
    assert (tmp_path / "resumed/loss_curve.png").is_file()
    assert (tmp_path / "resumed/val_1_after.png").is_file()
    with pytest.raises(ValueError, match="settings"):
        bad = args("bad", 3, checkpoint)
        bad.seed = 8
        run(bad)
    legacy = torch.load(checkpoint, weights_only=True)
    del legacy["objective_version"]
    legacy_path = tmp_path / "legacy.pt"
    torch.save(legacy, legacy_path)
    with pytest.raises(ValueError, match="incompatible training objective"):
        run(args("legacy", 2, legacy_path))
