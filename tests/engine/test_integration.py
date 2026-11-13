"""Opt-in real pretrained SqueezeNet test, strictly using existing cached weights."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from moodshift.engine.model_loader import pretrained_checkpoint_path
from moodshift.engine.neural_transfer import NeuralStyleTransfer
from moodshift.engine.settings import TransferSettings


@pytest.mark.integration
def test_real_pretrained_transfer(image_paths: tuple[Path, Path, Path]) -> None:
    """Actual pretrained features generate finite output influenced by reference."""
    if os.environ.get("MOODSHIFT_RUN_INTEGRATION") != "1":
        pytest.skip(
            "Set MOODSHIFT_RUN_INTEGRATION=1 to run cached pretrained inference"
        )
    if not pretrained_checkpoint_path().is_file():
        pytest.skip(
            "Official SqueezeNet checkpoint is not cached; test never downloads"
        )
    content, style, other = image_paths
    before = [path.read_bytes() for path in image_paths]
    engine = NeuralStyleTransfer(device="cpu", allow_download=False)
    settings = TransferSettings(steps=8, max_size=48, style_strength=1)
    output = engine.transfer(content, style, settings)
    different = engine.transfer(content, other, settings)
    array = np.asarray(output)
    assert output.mode == "RGB" and output.size == (48, 40)
    assert np.isfinite(array).all()
    with Image.open(content) as image:
        content_delta = np.abs(array.astype(float) - np.asarray(image)).mean()
    reference_delta = np.abs(array.astype(float) - np.asarray(different)).mean()
    assert content_delta > 0.1
    assert reference_delta > 0.1
    with Image.open(style) as reference:
        assert output.size != reference.size  # output follows content geometry
    assert [path.read_bytes() for path in image_paths] == before
    assert engine._extractor is not None
    assert all(not p.requires_grad for p in engine._extractor.parameters())
    assert not engine._extractor.training
    print(
        f"Real SqueezeNet CPU transfer: content delta={content_delta:.4f}, "
        f"reference-change delta={reference_delta:.4f}, torch={torch.__version__}"
    )
