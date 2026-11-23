"""Dataset tests use temporary synthetic fixtures, never training photographs."""

import shutil
from pathlib import Path

import pytest
import torch
from PIL import Image

from moodshift.engine.dataset import (
    PairedImageDataset,
    inspect_dataset,
    make_data_loader,
)


def populate(root: Path) -> None:
    """Create distinguishable images in each split and role."""
    for split_index, split in enumerate(("train", "val")):
        for role_index, role in enumerate(("content", "style")):
            directory = root / split / role
            directory.mkdir(parents=True)
            for index in range(5):
                value = 20 + 80 * split_index + 20 * role_index + index * 5
                image = Image.new("L", (28 + index, 40), value)
                image.save(directory / f"{index}.png")


@pytest.mark.parametrize("size", [96, 128])
def test_loading_and_batching(tmp_path: Path, size: int) -> None:
    """Load grayscale, alpha, JPEG and WebP as square RGB float tensors."""
    populate(tmp_path)
    Image.new("RGBA", (40, 20), (255, 0, 0, 0)).save(
        tmp_path / "train/content/alpha.png"
    )
    Image.new("RGB", (30, 40), "red").save(tmp_path / "train/style/photo.jpg")
    Image.new("RGB", (30, 40), "blue").save(tmp_path / "train/style/photo.webp")
    dataset = PairedImageDataset(tmp_path, "train", size)
    image, _ = dataset[0]
    assert image.shape == (3, size, size)
    assert torch.equal(image[0], image[1]) and torch.equal(image[1], image[2])
    assert dataset[-1][0].min() == 1  # alpha composites over white
    content, style = next(
        iter(make_data_loader(tmp_path, "train", size=size, batch_size=3))
    )
    assert content.shape == style.shape == (3, 3, size, size)
    assert 0 <= content.min() <= content.max() <= 1


def test_corrupt_missing_formats_and_duplicates(tmp_path: Path) -> None:
    """Reject corrupt or mislabeled images and report byte-identical split leakage."""
    assert len(inspect_dataset(tmp_path).problems) == 4
    with pytest.raises(ValueError, match="at least one valid"):
        PairedImageDataset(tmp_path, "train")
    populate(tmp_path)
    (tmp_path / "train/content/bad.png").write_bytes(b"broken")
    (tmp_path / "train/content/notes.txt").write_text("hello")
    shutil.copyfile(tmp_path / "train/content/0.png", tmp_path / "val/style/copy.png")
    shutil.copyfile(
        tmp_path / "train/content/1.png", tmp_path / "train/content/wrong.jpg"
    )
    report = inspect_dataset(tmp_path)
    assert len(report.files["train", "content"]) == 5
    assert len(report.problems) == 3
    assert len(report.duplicates) == 1
    assert not report.ready
    assert len(PairedImageDataset(tmp_path, "train")) == 5


def test_shuffle_and_split_isolation(tmp_path: Path) -> None:
    """Repeat the same seed and keep validation paths and data separate."""
    populate(tmp_path)

    def sequence(seed: int) -> torch.Tensor:
        batches = make_data_loader(tmp_path, "train", batch_size=2, seed=seed)
        return torch.cat([content[:, 0, 0, 0] for content, _ in batches])

    assert torch.equal(sequence(5), sequence(5))
    assert not torch.equal(sequence(5), sequence(6))
    train = PairedImageDataset(tmp_path, "train")
    val = PairedImageDataset(tmp_path, "val")
    assert set(train.content).isdisjoint(val.content)
    assert set(train.style).isdisjoint(val.style)
    assert train[0][0].mean() < val[0][0].mean()


def test_invalid_options(tmp_path: Path) -> None:
    """Fail clearly for unsupported configuration."""
    with pytest.raises(ValueError, match="split"):
        PairedImageDataset(tmp_path, "test")
    with pytest.raises(ValueError, match="size"):
        PairedImageDataset(tmp_path, "train", size=64)
    with pytest.raises(ValueError, match="batch_size"):
        make_data_loader(tmp_path, "train", batch_size=0)


def test_ready_counts_and_all_supported_formats(tmp_path: Path) -> None:
    """Starter thresholds apply separately to all four directories."""
    populate(tmp_path)
    for role in ("content", "style"):
        directory = tmp_path / "train" / role
        for index, extension in enumerate(("jpg", "jpeg", "webp", "png", "png")):
            Image.new("RGB", (25, 31), (index * 30, 100, 20)).save(
                directory / f"extra{index}.{extension}"
            )
    report = inspect_dataset(tmp_path)
    assert report.ready
    assert len(report.files["train", "content"]) == 10
    assert len(report.files["val", "style"]) == 5
