"""Training evidence and trusted experimental inference tests."""

import hashlib
from pathlib import Path

import pytest
import torch
from PIL import Image

from moodshift.ui import experiments
from moodshift.ui.app import create_app
from moodshift.ui.components import CSS
from moodshift.ui.controller import UiController
from moodshift.ui.experiments import (
    ExperimentError,
    experiment_info,
    infer,
    load_checkpoint,
)


def test_missing_and_invalid_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing or unrecognized weights never yield a substitute model."""
    info, rows, graph, samples, available = experiment_info(tmp_path)
    assert "not available" in info and not rows and graph is None and not samples
    assert not available
    folder = tmp_path / "checkpoints" / experiments.RUN_NAME
    folder.mkdir(parents=True)
    path = folder / "checkpoint.pt"
    torch.save({"architecture": "wrong"}, path)
    with pytest.raises(ExperimentError, match="trusted"):
        load_checkpoint(tmp_path)
    monkeypatch.setattr(
        experiments, "CHECKPOINT_SHA256", hashlib.sha256(path.read_bytes()).hexdigest()
    )
    with pytest.raises(ExperimentError, match="architecture"):
        load_checkpoint(tmp_path)
    with pytest.raises(ExperimentError, match="Upload both"):
        infer(None, None, experiments.RUN_NAME, tmp_path, tmp_path)


def test_ui_has_separate_training_tab() -> None:
    """The original Studio and isolated experimental callback both exist."""
    controller = UiController()
    app = create_app(controller)
    try:
        tabs = [
            c["props"].get("label")
            for c in app.config["components"]
            if c["type"] == "tabitem"
        ]
        assert "Studio" in tabs and "Training & Experiments" in tabs
        assert any(c["type"] == "dataframe" for c in app.config["components"])
        assert any(fn.api_name == "generate_experimental" for fn in app.fns.values())
        assert any(fn.api_name == "generate" for fn in app.fns.values())
        control = next(
            c
            for c in app.config["components"]
            if c["props"].get("label") == "Decoder color-change amplification"
        )
        assert control["props"]["value"] == 1.5
        assert control["props"]["minimum"] == 1.0
        assert control["props"]["maximum"] == 2.0
        for element in ("experimental-result", "experimental-comparison"):
            component = next(
                c
                for c in app.config["components"]
                if c["props"].get("elem_id") == element
            )
            assert component["props"]["height"] == 384
        assert "#experimental-result img" in CSS
        assert "image-rendering: pixelated" in CSS
    finally:
        app.close()
        controller.close()


def test_actual_checkpoint_inference_if_available(tmp_path: Path) -> None:
    """Run real CPU inference when the pinned local checkpoint is present."""
    try:
        checkpoint = load_checkpoint()
    except ExperimentError:
        pytest.skip("Pinned local experiment is unavailable")
    from moodshift.engine.model_loader import pretrained_checkpoint_path

    if not pretrained_checkpoint_path().is_file():
        pytest.skip("Official cached encoder is unavailable")
    info, rows, graph, samples, available = experiment_info()
    assert available and len(rows) == checkpoint["epoch"] and graph
    if not samples:
        assert "image is unavailable" in info
    content = tmp_path / "content.png"
    style = tmp_path / "style.png"
    Image.new("RGB", (120, 96), (70, 120, 200)).save(content)
    Image.new("RGB", (96, 120), (220, 100, 50)).save(style)
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        output, comparison, download = infer(
            str(content), str(style), experiments.RUN_NAME, tmp_path
        )
    finally:
        torch.set_num_threads(previous_threads)
    assert output.mode == "RGB" and output.size == (96, 96)
    assert comparison[0].size == comparison[1].size == output.size
    assert Path(download).is_file()
    with Image.open(download) as png:
        assert png.size == output.size
        assert png.convert("RGB").tobytes() == output.tobytes()
    high, pair, high_download, native, native_download = (
        experiments.infer_with_reconstruction(
            str(content), str(style), experiments.RUN_NAME, tmp_path
        )
    )
    assert native.tobytes() == output.tobytes()
    assert high.size == pair[0].size == pair[1].size == (96, 96)
    with Image.open(high_download) as png:
        assert png.convert("RGB").tobytes() == high.tobytes()
    with Image.open(native_download) as png:
        assert png.convert("RGB").tobytes() == output.tobytes()


def test_held_out_gallery_requires_matching_originals(tmp_path: Path) -> None:
    """Only an exact matching original may be used for a saved decoder result."""
    from moodshift.imaging.decoder_reconstruction import decoder_input_from_original

    directory = tmp_path / "checkpoints" / experiments.RUN_NAME
    directory.mkdir(parents=True)
    for role, color in (("content", (100, 120, 140)), ("style", (190, 100, 80))):
        folder = tmp_path / "data" / "val" / role
        folder.mkdir(parents=True)
        original = Image.new("RGB", (240, 160), color)
        original.save(folder / "original.png")
        decoder_input_from_original(original).save(directory / f"val_1_{role}.png")
    Image.new("RGB", (96, 96), (110, 120, 140)).save(directory / "val_1_guided.png")
    samples, messages = experiments.held_out_samples(tmp_path)
    assert len(samples) == 3 and samples[0][0].size == samples[2][0].size == (160, 160)
    assert any("Validation 2" in message for message in messages)
    Image.new("RGB", (240, 160), (1, 2, 3)).save(
        tmp_path / "data" / "val" / "content" / "original.png"
    )
    samples, messages = experiments.held_out_samples(tmp_path)
    assert not samples and any("Validation 1" in message for message in messages)
