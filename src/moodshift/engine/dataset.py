"""Offline dataset inspection and paired image loading for decoder training."""

import argparse
import hashlib
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from moodshift.engine.tensors import image_to_tensor
from moodshift.imaging.exceptions import ImageError
from moodshift.imaging.io import load_image

EXTENSIONS = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".webp": "WEBP"}
SPLITS = ("train", "val")
ROLES = ("content", "style")
# Practical starting targets, not evidence of model quality or dataset suitability.
MINIMUMS = {"train": 10, "val": 2}


@dataclass(frozen=True)
class DatasetReport:
    """Validated paths, rejected files, missing directories, and cross-split copies."""

    files: dict[tuple[str, str], tuple[Path, ...]]
    problems: tuple[str, ...]
    duplicates: tuple[tuple[Path, Path], ...]

    @property
    def ready(self) -> bool:
        """Indicate whether the starter counts and basic integrity checks pass."""
        return (
            not self.problems
            and not self.duplicates
            and all(
                len(self.files[split, role]) >= MINIMUMS[split]
                for split in SPLITS
                for role in ROLES
            )
        )


def inspect_dataset(root: str | Path) -> DatasetReport:
    """Scan four directories, checking actual formats, decodability, and file hashes.

    Unsupported files are reported; subdirectories are not traversed. Hashes
    detect byte-identical copies, not visually similar or re-encoded images.
    """
    root = Path(root)
    files: dict[tuple[str, str], tuple[Path, ...]] = {}
    problems: list[str] = []
    hashes: dict[str, list[tuple[str, Path]]] = {}
    duplicates: list[tuple[Path, Path]] = []
    for split in SPLITS:
        for role in ROLES:
            directory = root / split / role
            valid: list[Path] = []
            if not directory.is_dir():
                problems.append(f"Missing folder: {directory}")
                files[split, role] = ()
                continue
            for path in sorted(directory.iterdir()):
                if not path.is_file():
                    continue
                expected = EXTENSIONS.get(path.suffix.lower())
                if expected is None:
                    problems.append(f"Unsupported extension: {path}")
                    continue
                try:
                    image = load_image(path)
                    image.close()
                    with Image.open(path) as source:
                        if source.format != expected:
                            raise ValueError(
                                f"expected {expected}, found {source.format}"
                            )
                    with path.open("rb") as stream:
                        digest = hashlib.file_digest(stream, "sha256").hexdigest()
                except (ImageError, OSError, ValueError) as error:
                    problems.append(f"Invalid image {path}: {error}")
                    continue
                for previous_split, previous_path in hashes.get(digest, []):
                    if previous_split != split:
                        duplicates.append((previous_path, path))
                hashes.setdefault(digest, []).append((split, path))
                valid.append(path)
            files[split, role] = tuple(valid)
    return DatasetReport(files, tuple(problems), tuple(duplicates))


class PairedImageDataset(Dataset[tuple[Tensor, Tensor]]):
    """Pair each content image with a seeded, cycling selection of style images."""

    def __init__(self, root: str | Path, split: str, size: int = 96, seed: int = 0):
        """Validate one split and prepare fixed, reproducible content/style pairs."""
        if split not in SPLITS:
            raise ValueError("split must be train or val")
        if type(size) is not int or size not in (96, 128):
            raise ValueError("size must be 96 or 128")
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("seed must be an integer between 0 and 2**63 - 1")
        report = inspect_dataset(root)
        self.content = report.files[split, "content"]
        styles = list(report.files[split, "style"])
        if not self.content or not styles:
            raise ValueError(
                f"{split} requires at least one valid content and style image"
            )
        # Rejected images remain visible to callers instead of silently disappearing.
        self.problems = report.problems
        self.duplicates = report.duplicates
        random.Random(seed).shuffle(styles)
        self.style = tuple(styles[i % len(styles)] for i in range(len(self.content)))
        self.size = size

    def __len__(self) -> int:
        """Return the number of content images in the split."""
        return len(self.content)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """Load one pair as RGB float tensors; report files changed after scanning."""
        return self.load(self.content[index]), self.load(self.style[index])

    def load(self, path: Path) -> Tensor:
        """Load an image using the same preprocessing for both references."""
        try:
            image = load_image(path)
            resized = ImageOps.fit(
                image, (self.size, self.size), method=Image.Resampling.LANCZOS
            )
            return image_to_tensor(resized, torch.device("cpu")).squeeze(0)
        except (ImageError, OSError) as error:
            raise ValueError(f"Cannot load dataset image {path}: {error}") from error


class PairedReferenceDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """Add a reproducible, byte-distinct second reference from the same split."""

    def __init__(self, root: str | Path, split: str, size: int = 96, seed: int = 0):
        """Choose a second reference reproducibly from the requested split."""
        self.base = PairedImageDataset(root, split, size, seed)
        styles = list(inspect_dataset(root).files[split, "style"])
        hashes = {path: hashlib.sha256(path.read_bytes()).digest() for path in styles}
        rng = random.Random(seed)
        self.second_style = tuple(
            rng.choice(candidates)
            for first in self.base.style
            for candidates in ([p for p in styles if hashes[p] != hashes[first]],)
            if candidates
        )
        if len(self.second_style) != len(self.base):
            raise ValueError(f"{split} needs two byte-distinct style references")

    def __len__(self) -> int:
        """Return the number of content images."""
        return len(self.base)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        """Load content and its two references with the same preprocessing."""
        content, style = self.base[index]
        return content, style, self.base.load(self.second_style[index])


def make_paired_reference_loader(
    root: str | Path, split: str, *, size: int = 96, batch_size: int = 1, seed: int = 0
) -> DataLoader:
    """Use the original seeded first-reference order and same-split second styles."""
    if type(batch_size) is not int or not 1 <= batch_size <= 32:
        raise ValueError("batch_size must be an integer between 1 and 32")
    return DataLoader(
        PairedReferenceDataset(root, split, size, seed),
        batch_size=batch_size,
        shuffle=split == "train",
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )


def make_data_loader(
    root: str | Path, split: str, *, size: int = 96, batch_size: int = 4, seed: int = 0
) -> DataLoader:
    """Create a CPU loader with seeded train shuffle and deterministic validation."""
    if type(batch_size) is not int or not 1 <= batch_size <= 32:
        raise ValueError("batch_size must be an integer between 1 and 32")
    dataset = PairedImageDataset(root, split, size, seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == "train",
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )


def main() -> int:
    """Validate a dataset or inspect one batch without loading a neural model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--smoke", action="store_true", help="load one batch per split")
    parser.add_argument("--size", type=int, choices=(96, 128), default=96)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    report = inspect_dataset(args.data)
    for split in SPLITS:
        for role in ROLES:
            count = len(report.files[split, role])
            print(
                f"{split}/{role}: {count} valid images "
                f"(starter target: {MINIMUMS[split]})"
            )
    for problem in report.problems:
        print(f"WARNING: {problem}")
    for first, second in report.duplicates:
        print(f"WARNING: byte-identical images across splits: {first} and {second}")
    print("Dataset checks: " + ("starter targets met" if report.ready else "not ready"))
    if args.smoke:
        for split in SPLITS:
            try:
                loader = make_data_loader(
                    args.data,
                    split,
                    size=args.size,
                    batch_size=args.batch_size,
                    seed=args.seed,
                )
                content, style = next(iter(loader))
                print(
                    f"{split} batch: content {tuple(content.shape)}, "
                    f"style {tuple(style.shape)}"
                )
            except (ValueError, OSError) as error:
                print(f"{split} smoke test failed: {error}")
                return 1
    return 0 if report.ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
