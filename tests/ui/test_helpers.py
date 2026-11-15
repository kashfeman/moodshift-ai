"""Mode behavior and typed UI controls without any server or model download."""

from dataclasses import replace

import pytest

from moodshift.ui.helpers import friendly_error, mode_presentation
from moodshift.ui.state import UiParameters


@pytest.mark.parametrize("mode", ["grade", "neural", "hybrid"])
@pytest.mark.parametrize("reference", [False, True])
def test_mode_controls(mode: str, reference: bool) -> None:
    """Mode presentation follows actual backend routing and processing stages."""
    view = mode_presentation(mode, reference, 70)
    assert view.neural_visible == (mode != "grade")
    assert view.preset_enabled == (mode != "neural" or not reference)
    assert view.grain_enabled == (mode != "neural")
    if mode != "grade":
        assert ("uploaded reference" if reference else "synthetic") in view.explanation
    assert "Zero style intensity" in mode_presentation(mode, reference, 0).explanation


def test_zero_neural_strength_explanation() -> None:
    """Zero neural strength has distinct neural-only and hybrid behavior."""
    assert "unchanged" in mode_presentation("neural", False, 70, 0).explanation
    assert (
        "Only the selected photographic grade"
        in mode_presentation("hybrid", True, 70, 0).explanation
    )


def test_hidden_controls_and_reference_routing() -> None:
    """Hidden neural values do not leak into grade-only requests."""
    controls = UiParameters(
        "content.png",
        reference="reference.png",
        quality="invalid",
        preservation=-1,
        steps=-1,
        custom_steps=True,
        style_strength=-1,
    )
    settings = controls.settings()
    assert settings.mode == "grade" and settings.steps is None
    assert settings.neural_style_strength == 1
    assert controls.selection() == ("cinematic", None)
    neural = replace(
        controls,
        mode="neural",
        quality="preview",
        preservation=0.7,
        custom_steps=False,
        style_strength=0.4,
    )
    assert neural.selection() == (None, "reference.png")
    assert neural.settings().neural_settings().style_strength == 0.4
    hybrid = replace(neural, mode="hybrid")
    assert hybrid.selection() == ("cinematic", "reference.png")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mode", "bad"),
        ("intensity", float("nan")),
        ("intensity", 101),
        ("seed", -1),
        ("device", "gpu"),
        ("offline", 1),
    ],
)
def test_invalid_ui_controls(field: str, value: object) -> None:
    """Invalid form submissions fail through the service's real setting validation."""
    with pytest.raises(ValueError):
        UiParameters("content.png", **{field: value}).settings()


def test_safe_error_messages() -> None:
    """Unexpected filesystem/device details are not copied into user-facing text."""
    secret = "C:/private/user/image.png"
    for error in (
        OSError(secret),
        RuntimeError(secret),
        ImportError(secret),
        Exception(secret),
    ):
        assert secret not in friendly_error(error)
    assert "memory" in friendly_error(MemoryError()).lower()
    assert (
        friendly_error(ValueError("Upload a content image."))
        == "Upload a content image."
    )
