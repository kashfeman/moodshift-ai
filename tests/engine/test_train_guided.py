"""Configuration, reproducibility and output isolation for guided experiments."""

import json
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image
from torchvision.models import squeezenet1_1

from moodshift.engine.dataset import inspect_dataset
from moodshift.engine.features import SqueezeNetFeatures
from moodshift.engine.guided_decoder import BASELINE_WEIGHTS, STRONGER_STYLE_WEIGHTS
from moodshift.engine.train_guided import dataset_fingerprint, run


def test_comparable_run_metadata_without_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify pair ordering and initialization with a mocked, no-update step."""
    import moodshift.engine.train_guided as module

    data = tmp_path / "data"
    for split in ("train", "val"):
        for role in ("content", "style"):
            folder = data / split / role
            folder.mkdir(parents=True)
            for i in range(10 if split == "train" else 2):
                # Distinct across splits so the real split validator can run.
                Image.new(
                    "RGB",
                    (32, 35),
                    (
                        i * 12,
                        40 if split == "train" else 140,
                        30 if role == "content" else 100,
                    ),
                ).save(folder / f"{i}.png")
    fake_checkpoint = tmp_path / "encoder.pth"
    fake_checkpoint.write_bytes(b"unit test model boundary")
    monkeypatch.setattr(module, "pretrained_checkpoint_path", lambda: fake_checkpoint)
    monkeypatch.setattr(
        module,
        "load_feature_extractor",
        lambda device, allow_download: SqueezeNetFeatures(
            squeezenet1_1(weights=None).features
        ),
    )
    seen = []

    def no_update(encoder, decoder, optimizer, content, style, weights):
        seen.append((content.mean().item(), style.mean().item(), weights))
        return {
            "loss": 1.0,
            "content_loss": 0.2,
            "style_loss": 0.3,
            "reconstruction_loss": 0.1,
            "edge_loss": 0.1,
        }

    monkeypatch.setattr(module, "guided_step", no_update)
    monkeypatch.setattr(module, "conditioned_step", no_update)
    original_threads = torch.get_num_threads()
    try:
        run(data, tmp_path / "baseline", epochs=1, seed=4)
        first = seen[:]
        seen.clear()
        run(
            data,
            tmp_path / "strong",
            epochs=1,
            seed=4,
            weights=STRONGER_STYLE_WEIGHTS,
            preset="stronger-style",
        )
        run(
            data,
            tmp_path / "conditioned",
            epochs=1,
            seed=4,
            architecture="style-film-v1",
        )
    finally:
        torch.set_num_threads(original_threads)
    a = json.loads((tmp_path / "baseline/experiment.json").read_text())
    b = json.loads((tmp_path / "strong/experiment.json").read_text())
    assert [(x, y) for x, y, _ in first] == [(x, y) for x, y, _ in seen[:10]]
    assert all(w == BASELINE_WEIGHTS for _, _, w in first)
    assert all(w == STRONGER_STYLE_WEIGHTS for _, _, w in seen[:10])
    conditioned = json.loads((tmp_path / "conditioned/experiment.json").read_text())
    assert conditioned["architecture"] == "style-film-v1"
    assert conditioned["shared_baseline_initial_sha256"] == a["initial_decoder_sha256"]
    assert (
        torch.load(tmp_path / "conditioned/checkpoint.pt", weights_only=True)[
            "architecture"
        ]
        == "style-film-v1"
    )
    for key in (
        "seed",
        "epochs",
        "dataset_sha256",
        "initial_decoder_sha256",
        "encoder_checkpoint_sha256",
        "learning_rate",
        "pairing",
    ):
        assert a[key] == b[key]
    assert b["loss_weights"] == {"content": 1, "style": 3, "identity": 5, "edge": 1}
    saved = torch.load(tmp_path / "strong/checkpoint.pt", weights_only=True)
    assert saved["experiment"] == b
    assert (tmp_path / "strong/checkpoint_epoch_001.pt").is_file()
    assert (tmp_path / "strong/losses_epoch_001.csv").is_file()
    old = (tmp_path / "strong/checkpoint.pt").read_bytes()
    with pytest.raises(FileExistsError):
        run(data, tmp_path / "strong", epochs=1, seed=4)
    assert old == (tmp_path / "strong/checkpoint.pt").read_bytes()
    assert dataset_fingerprint(data, inspect_dataset(data).files) == a["dataset_sha256"]


def test_cli_preset_and_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI applies preset values and validated explicit overrides before running."""
    import moodshift.engine.train_guided as module

    calls = []
    monkeypatch.setattr(module, "run", lambda *args: calls.append(args))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_guided",
            "--output",
            "unused",
            "--loss-preset",
            "stronger-style",
            "--edge-weight",
            "0.5",
        ],
    )
    module.main()
    assert calls[0][4].style == 3
    assert calls[0][4].identity == 5
    assert calls[0][4].edge == 0.5
    monkeypatch.setattr(
        sys, "argv", ["train_guided", "--output", "unused", "--style-weight", "nan"]
    )
    with pytest.raises(SystemExit):
        module.main()
    assert len(calls) == 1
