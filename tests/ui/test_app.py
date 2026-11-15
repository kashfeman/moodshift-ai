"""Build and callback tests use Gradio itself without launching a browser/server."""

from collections.abc import Iterator
from dataclasses import fields
from pathlib import Path

import gradio as gr
import pytest
from PIL import Image

from moodshift.ui.app import create_app, main
from moodshift.ui.callbacks import make_generate
from moodshift.ui.controller import UiController
from moodshift.ui.state import UiParameters


@pytest.fixture
def controller() -> Iterator[UiController]:
    """Create/clean an adapter without loading any neural model."""
    value = UiController()
    yield value
    value.close()


def test_blocks_build_and_single_generation_queue(controller: UiController) -> None:
    """All three modes and the comparison component exist with serialized generation."""
    app = create_app(controller)
    try:
        assert app.title == "MoodShift AI" and not controller._services
        radios = [
            component
            for component in app.config["components"]
            if component["type"] == "radio"
        ]
        assert [choice[1] for choice in radios[0]["props"]["choices"]] == [
            "grade",
            "neural",
            "hybrid",
        ]
        assert any(c["type"] == "imageslider" for c in app.config["components"])
        generation = next(fn for fn in app.fns.values() if fn.api_name == "generate")
        assert generation.concurrency_limit == 1
        assert generation.concurrency_id == "moodshift-generation"
        assert generation.trigger_mode == "once"
    finally:
        app.close()


def run_callback(controller: UiController, parameters: UiParameters) -> list[tuple]:
    """Consume streaming callback with explicit progress injection for unit tests."""
    values = [getattr(parameters, field.name) for field in fields(UiParameters)]
    return list(
        make_generate(controller)(
            *values,
            request=gr.Request(session_hash="test"),
            progress=lambda *args, **kwargs: None,
        )
    )


def test_callback_success_and_failure_reenable_button(
    controller: UiController, tmp_path: Path
) -> None:
    """Button and output state settle cleanly on success and missing-input errors."""
    path = tmp_path / "source.png"
    Image.new("RGB", (40, 32), (80, 120, 160)).save(path)
    events = run_callback(controller, UiParameters(str(path)))
    assert events[0][-1].interactive is False
    assert events[-1][-1].interactive is True
    assert Path(events[-1][2]).is_file()
    assert events[-1][4]["actual_engine"] == "numpy-opencv"
    assert events[-1][1][0].size == events[-1][1][1].size
    failure = run_callback(controller, UiParameters(None))
    assert failure[-1][-1].interactive is True
    assert all(value is None for value in failure[-1][:3])
    assert "Upload a content image" in failure[-1][-2]
    assert not controller._sessions


def test_launch_port_validation() -> None:
    """Invalid ports never launch a server."""
    with pytest.raises(SystemExit) as error:
        main(["--port", "0"])
    assert error.value.code == 2
