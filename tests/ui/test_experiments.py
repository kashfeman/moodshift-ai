"""Training evidence and trusted experimental inference tests."""

import hashlib
from pathlib import Path

import pytest
import torch
from PIL import Image

from moodshift.ui import experiments
from moodshift.ui.app import create_app
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
    assert Path(download).is_file() and Image.open(download).size == output.size
