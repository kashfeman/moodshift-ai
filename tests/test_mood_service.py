"""Service unit tests mock only the neural boundary, never claim real inference."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from moodshift.engine.settings import TransferSettings
from moodshift.imaging.exceptions import InvalidImageError
from moodshift.imaging.io import load_image
from moodshift.imaging.reference_images import synthetic_scene
from moodshift.imaging.transforms import resize_to_fit
from moodshift.presets import PRESETS
from moodshift.services.demo import main
from moodshift.services.models import MoodSettings
from moodshift.services.mood_transfer import MoodTransferService


class RecordingNeuralBackend:
    """Boundary test double with deterministic output, labeled as injected backend."""

    device = "test-cpu"

    def __init__(self) -> None:
        """Record actual calls and reference pixels seen inside their lifetime."""
        self.calls: list[tuple[Path, bytes, TransferSettings]] = []

    def transfer(
        self,
        content_path: str | Path,
        style_path: str | Path,
        settings: TransferSettings | None = None,
    ) -> Image.Image:
        """Return a known RGB image, without simulating a pretrained network."""
        assert settings is not None
        style = load_image(style_path)
        self.calls.append((Path(style_path), style.tobytes(), settings))
        content = resize_to_fit(
            load_image(content_path), (settings.max_size, settings.max_size)
        )
        return Image.new("RGB", content.size, (140, 100, 170))


@pytest.fixture
def content_path(tmp_path: Path) -> Path:
    """Write a generated scene without external test assets."""
    path = tmp_path / "content.png"
    synthetic_scene().save(path)
    return path


@pytest.mark.parametrize("preset_id", [p.id for p in PRESETS])
def test_grade_only_and_metadata(content_path: Path, preset_id: str) -> None:
    """Named grades preserve dimensions and truthfully report no neural execution."""
    backend = RecordingNeuralBackend()
    original = content_path.read_bytes()
    result = MoodTransferService(neural_engine=backend).transfer(
        content_path, preset_id=preset_id, settings=MoodSettings(intensity=100)
    )
    assert not backend.calls
    assert result.image.size == (320, 240)
    facts = result.metadata
    assert facts.actual_engine == "numpy-opencv" and facts.device == "cpu"
    assert facts.grading_applied and not facts.neural_applied
    assert facts.reference_source == "none" and facts.settings.neural_steps == 0
    assert facts.preset_id == preset_id and facts.neural_size is None
    assert content_path.read_bytes() == original


@pytest.mark.parametrize("mode", ["grade", "hybrid", "neural"])
def test_zero_intensity(content_path: Path, mode: str) -> None:
    """Every mode returns exact normalized original pixels and invokes no model."""
    backend = RecordingNeuralBackend()
    result = MoodTransferService(neural_engine=backend).transfer(
        content_path,
        preset_id="cinematic",
        settings=MoodSettings(mode=mode, intensity=0, max_size=32),
    )
    assert result.image.tobytes() == load_image(content_path).tobytes()
    assert result.image.size == (320, 240) and not backend.calls
    assert result.metadata.actual_engine == "none"
    assert result.metadata.reference_source == "none"
    assert result.metadata.settings.neural_steps == 0
    assert not result.metadata.neural_applied and not result.metadata.grading_applied


@pytest.mark.parametrize("mode", ["hybrid", "neural"])
def test_neural_modes_use_backend(content_path: Path, mode: str) -> None:
    """Positive neural modes consume synthetic references and use resolved quality."""
    backend = RecordingNeuralBackend()
    service = MoodTransferService(neural_engine=backend)
    result = service.transfer(
        content_path,
        preset_id="sunset",
        settings=MoodSettings(mode=mode, quality="preview"),
    )
    assert len(backend.calls) == 1
    path, reference_pixels, controls = backend.calls[0]
    assert reference_pixels and not path.exists()  # Temporary reference was cleaned up.
    assert controls.steps == 20 and controls.max_size == 128
    assert result.image.size == (320, 240)
    assert result.metadata.neural_size == (128, 96)
    assert result.metadata.device == "test-cpu"
    assert result.metadata.actual_engine == "injected-neural-backend"
    assert result.metadata.neural_applied
    assert result.metadata.grading_applied == (mode == "hybrid")
    assert result.metadata.reference_source == "synthetic-preset"


def test_reference_and_explicit_size(content_path: Path, tmp_path: Path) -> None:
    """Apply user references without modification and honor explicit output bounds."""
    reference = tmp_path / "reference.png"
    Image.new("RGB", (48, 40), (30, 80, 190)).save(reference)
    before = reference.read_bytes()
    backend = RecordingNeuralBackend()
    service = MoodTransferService(neural_engine=backend)
    for mode in ("neural", "hybrid"):
        result = service.transfer(
            content_path,
            reference=reference,
            preset_id="dreamy" if mode == "hybrid" else None,
            settings=MoodSettings(mode=mode, max_size=64, steps=3),
        )
        assert result.image.size == (64, 48)
        assert result.metadata.reference_source == "user"
        assert result.metadata.settings.neural_steps == 3
        assert backend.calls[-1][0] == reference
    assert reference.read_bytes() == before
    grade = service.transfer(
        content_path, preset_id="vintage", settings=MoodSettings(max_size=64)
    )
    assert grade.image.size == (64, 48)
    assert len(backend.calls) == 2


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("mode", "fake"),
        ("quality", "ultra"),
        ("intensity", -1),
        ("intensity", 101),
        ("intensity", True),
        ("intensity", float("nan")),
        ("content_preservation", 2),
        ("max_size", 513),
        ("steps", 0),
        ("seed", -1),
        ("grain_enabled", 1),
    ],
)
def test_invalid_settings(name: str, value: object) -> None:
    """Invalid requests fail before any model can be loaded."""
    with pytest.raises(ValueError):
        MoodSettings(**{name: value})


def test_invalid_requests(content_path: Path, tmp_path: Path) -> None:
    """Reject ambiguity, unused references, missing presets, and invalid files."""
    service = MoodTransferService()
    for kwargs in (
        {},
        {"preset_id": "unknown"},
        {"settings": MoodSettings(mode="neural")},
        {"preset_id": "sunset", "reference": content_path},
        {
            "preset_id": "sunset",
            "reference": content_path,
            "settings": MoodSettings(mode="neural"),
        },
    ):
        with pytest.raises(ValueError):
            service.transfer(content_path, **kwargs)
    with pytest.raises(InvalidImageError):
        service.transfer(tmp_path / "missing.png", preset_id="sunset")
    with pytest.raises(InvalidImageError):
        service.transfer(
            content_path,
            reference=tmp_path / "missing.png",
            settings=MoodSettings(mode="neural", intensity=0),
        )


@pytest.mark.parametrize(
    ("quality", "steps", "size"),
    [("preview", 20, 128), ("balanced", 60, 256), ("high", 120, 384)],
)
def test_quality_defaults(quality: str, steps: int, size: int) -> None:
    """Service quality presets resolve to documented, bounded engine controls."""
    resolved = MoodSettings(quality=quality).neural_settings()
    assert resolved.steps == steps and resolved.max_size == size


def test_single_service_intensity_blend(content_path: Path) -> None:
    """Service intensity is applied exactly once to the complete grade."""
    service = MoodTransferService()
    original = np.asarray(load_image(content_path)).astype(float)
    full = service.transfer(
        content_path, preset_id="sunset", settings=MoodSettings(intensity=100)
    )
    halfway = service.transfer(
        content_path, preset_id="sunset", settings=MoodSettings(intensity=50)
    )
    expected = (original + np.asarray(full.image).astype(float)) / 2
    assert np.abs(np.asarray(halfway.image).astype(float) - expected).max() <= 0.5


def test_failed_neural_stage_propagates_and_cleans_reference(
    content_path: Path,
) -> None:
    """An inference failure leaves no temporary reference and returns no fallback."""

    class FailingBackend(RecordingNeuralBackend):
        def transfer(
            self,
            content_path: str | Path,
            style_path: str | Path,
            settings: TransferSettings | None = None,
        ) -> Image.Image:
            super().transfer(content_path, style_path, settings)
            raise RuntimeError("Test neural execution failed")

    backend = FailingBackend()
    service = MoodTransferService(neural_engine=backend)
    with pytest.raises(RuntimeError, match="neural execution failed"):
        service.transfer(
            content_path, preset_id="vintage", settings=MoodSettings(mode="hybrid")
        )
    assert backend.calls and not backend.calls[0][0].exists()


def test_grading_process_never_imports_torch(content_path: Path) -> None:
    """A fresh process can run the actual service with all torch imports forbidden."""
    code = """
import sys
class ForbidTorch:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'torchvision'):
            raise AssertionError('Grading must never import PyTorch')
sys.meta_path.insert(0, ForbidTorch())
from moodshift.services.mood_transfer import MoodTransferService
result = MoodTransferService().transfer(sys.argv[1], preset_id='cinematic')
assert result.metadata.actual_engine == 'numpy-opencv'
assert 'torch' not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", code, str(content_path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_service_cli(
    content_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """CLI exports real grading output, reports no neural work, and protects sources."""
    output = tmp_path / "result.png"
    args = [
        "--content",
        str(content_path),
        "--preset",
        "sunset",
        "--mode",
        "grade",
        "--intensity",
        "100",
        "--output",
        str(output),
    ]
    assert main(args) == 0
    assert "engine=numpy-opencv" in capsys.readouterr().out
    assert load_image(output).size == (320, 240)
    assert not np.array_equal(
        np.asarray(load_image(output)), np.asarray(load_image(content_path))
    )
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 1
