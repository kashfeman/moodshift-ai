"""Separate actual pretrained service smoke tests; no mock proves neural execution."""

import os
from pathlib import Path

import numpy as np
import pytest

from moodshift.engine.model_loader import pretrained_checkpoint_path
from moodshift.imaging.io import load_image
from moodshift.services.models import MoodSettings
from moodshift.services.mood_transfer import MoodTransferService


@pytest.mark.integration
@pytest.mark.parametrize("mode", ["neural", "hybrid"])
def test_real_mood_service(image_paths: tuple[Path, Path, Path], mode: str) -> None:
    """Use cached pretrained features and verify service execution metadata."""
    if os.environ.get("MOODSHIFT_RUN_INTEGRATION") != "1":
        pytest.skip("Opt-in cached pretrained service smoke test")
    if not pretrained_checkpoint_path().is_file():
        pytest.skip("No cached official weights; this test never downloads")
    content, reference, _ = image_paths
    before = [p.read_bytes() for p in image_paths]
    result = MoodTransferService(device="cpu", allow_download=False).transfer(
        content,
        preset_id="dreamy" if mode == "hybrid" else None,
        reference=None if mode == "hybrid" else reference,
        settings=MoodSettings(mode=mode, quality="preview", steps=4, intensity=75),
    )
    original = load_image(content)
    assert result.image.size == original.size
    difference = np.abs(
        np.asarray(result.image).astype(float) - np.asarray(original)
    ).mean()
    assert difference > 0.1
    assert np.isfinite(np.asarray(result.image)).all()
    assert result.metadata.actual_engine == "squeezenet1_1"
    assert result.metadata.device == "cpu" and result.metadata.neural_applied
    assert result.metadata.grading_applied == (mode == "hybrid")
    assert result.metadata.settings.neural_steps == 4
    assert result.metadata.reference_source == (
        "synthetic-preset" if mode == "hybrid" else "user"
    )
    assert [p.read_bytes() for p in image_paths] == before
    print(f"Real {mode} service: mean pixel change={difference:.3f}/255, device=cpu")
