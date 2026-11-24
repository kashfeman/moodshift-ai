"""Presentation adapters use the service boundary and export genuine RGB downloads."""

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from moodshift.imaging.io import load_image
from moodshift.services.models import MoodSettings
from moodshift.services.mood_transfer import MoodTransferService
from moodshift.ui.controller import UiController
from moodshift.ui.helpers import format_metadata
from moodshift.ui.state import UiParameters


@pytest.fixture
def content(tmp_path: Path) -> Path:
    """Generate a small source image for deterministic local grading."""
    path = tmp_path / "input.png"
    array = np.random.default_rng(3).integers(30, 220, (40, 48, 3), dtype=np.uint8)
    Image.fromarray(array).save(path)
    return path


@pytest.fixture
def controller() -> Iterator[UiController]:
    """Own and clean up only the adapter's isolated temporary directory."""
    value = UiController()
    yield value
    value.close()


def test_real_grade_and_unique_exports(controller: UiController, content: Path) -> None:
    """The real grade service supplies matching comparison images and safe PNG files."""
    original = content.read_bytes()
    stages = []
    controls = UiParameters(
        str(content), preset="sunset", intensity=80, limit_output=True, max_size=32
    )
    with controller.claim("session"):
        output = controller.generate(controls, stage=stages.append)
    assert output.image.size == (32, 27)
    assert output.comparison[0].size == output.comparison[1].size
    assert output.metadata["actual_engine"] == "numpy-opencv"
    assert output.metadata["settings"]["neural_steps"] == 0
    assert load_image(output.download).tobytes() == output.image.tobytes()
    assert content.read_bytes() == original
    second = controller.generate(controls, stage=stages.append)
    assert output.download != second.download
    assert stages[0] == "Validating inputs" and stages[-1].startswith("Preparing")


def test_mocked_service_boundary(controller: UiController, content: Path) -> None:
    """UI integration forwards typed controls and displays exactly the backend facts."""
    known = MoodTransferService().transfer(
        content, preset_id="cinematic", settings=MoodSettings(intensity=0)
    )
    backend = Mock()
    backend.transfer.return_value = known
    factory = Mock(return_value=backend)
    controller._factory = factory
    controls = UiParameters(
        str(content), mode="hybrid", custom_steps=True, steps=3, style_strength=0.4
    )
    result = controller.generate(controls, stage=lambda _: None)
    forwarded = backend.transfer.call_args.kwargs["settings"]
    assert forwarded.steps == 3 and forwarded.neural_style_strength == 0.4
    assert result.image.tobytes() == known.image.tobytes()
    assert result.metadata["actual_engine"] == "none"  # Never infer facts from UI mode.
    assert result.metadata["neural_applied"] is False
    controller.generate(controls, stage=lambda _: None)
    factory.assert_called_once_with(True, "auto")


def test_missing_and_invalid_uploads(
    controller: UiController, content: Path, tmp_path: Path
) -> None:
    """Missing, corrupt, and invalid controls are rejected before processing."""
    with pytest.raises(ValueError, match="Upload a content"):
        controller.generate(UiParameters(None), stage=lambda _: None)
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"broken")
    with pytest.raises(ValueError):
        controller.preview(str(broken))
    assert controller.preview(None) is None
    assert controller.preview(str(content)).size == (48, 40)


def test_session_guard_always_released(controller: UiController) -> None:
    """A duplicate is rejected, while failure releases the session for later work."""
    with pytest.raises(RuntimeError):
        with controller.claim("one"):
            with pytest.raises(ValueError, match="already generating"):
                with controller.claim("one"):
                    pass
            with controller.claim("two"):
                pass
            raise RuntimeError("test failure")
    with controller.claim("one"):
        pass


def test_metadata_uses_actual_execution(content: Path) -> None:
    """No-op facts never claim a device, neural optimization, or reference use."""
    result = MoodTransferService().transfer(
        content, preset_id="dreamy", settings=MoodSettings(mode="hybrid", intensity=0)
    )
    summary, details = format_metadata(result.metadata)
    assert "**Engine:** none" in summary and "**Device:** Not used" in summary
    assert "None applied" in summary and "**Neural steps:** 0" in summary
    assert details["settings"]["intensity"] == 0


@pytest.mark.parametrize("mode", ["neural", "hybrid"])
def test_zero_neural_strength_skips_model(content: Path, mode: str) -> None:
    """Zero neural strength skips inference with truthful hybrid/no-op facts."""
    backend = Mock()
    service = MoodTransferService(neural_engine=backend)
    result = service.transfer(
        content,
        preset_id="cinematic",
        settings=MoodSettings(mode=mode, neural_style_strength=0, max_size=32),
    )
    backend.transfer.assert_not_called()
    assert not result.metadata.neural_applied
    assert result.metadata.settings.neural_steps == 0
    assert result.metadata.reference_source == "none"
    assert result.metadata.grading_applied == (mode == "hybrid")
    if mode == "neural":
        assert result.image.tobytes() == load_image(content).tobytes()


def test_neural_strength_pass_through_and_bounds() -> None:
    """New optional service control preserves defaults and uses engine validation."""
    assert MoodSettings().neural_settings().style_strength == 1
    assert (
        MoodSettings(neural_style_strength=0.4).neural_settings().style_strength == 0.4
    )
    for strength in (-1, 2, float("nan"), True):
        with pytest.raises(ValueError):
            MoodSettings(neural_style_strength=strength)


def test_reference_boost_after_neural_and_missing_reference(
    controller: UiController, content: Path, tmp_path: Path
) -> None:
    """Apply the boost only after a real neural result and preserve PNG export."""
    from dataclasses import replace

    from moodshift.services.models import MoodResult

    original = load_image(content)
    known = MoodTransferService().transfer(
        content, preset_id="cinematic", settings=MoodSettings(intensity=0)
    )
    neural_result = MoodResult(
        original, replace(known.metadata, neural_applied=True, actual_engine="test")
    )
    backend = Mock()
    backend.transfer.return_value = neural_result
    controller._factory = Mock(return_value=backend)
    reference = tmp_path / "warm.png"
    Image.new("RGB", (40, 40), (230, 100, 30)).save(reference)
    common = {"mode": "neural", "boost_enabled": True, "boost_strength": 100}
    no_ref = controller.generate(
        UiParameters(str(content), **common), stage=lambda _: None
    )
    assert no_ref.image.tobytes() == original.tobytes()
    boosted = controller.generate(
        UiParameters(str(content), boost_reference=str(reference), **common),
        stage=lambda _: None,
    )
    assert boosted.metadata["reference_mood_boost"]["applied"]
    assert boosted.image.tobytes() != original.tobytes()
    assert load_image(boosted.download).tobytes() == boosted.image.tobytes()
    assert backend.transfer.call_count == 2
