"""Streaming generation callback with predictable busy, success, and error states."""

import logging
from collections.abc import Callable, Iterator

import gradio as gr

from moodshift.ui.controller import UiController
from moodshift.ui.helpers import friendly_error
from moodshift.ui.state import UiParameters


def make_generate(controller: UiController) -> Callable[..., Iterator[tuple]]:
    """Bind one controller and Gradio progress reporter without global app state."""
    progress = gr.Progress()

    def run(
        content: str | None,
        mode: str,
        preset: str,
        reference: str | None,
        intensity: float,
        quality: str,
        preservation: float,
        custom_steps: bool,
        steps: int,
        limit_output: bool,
        max_size: int,
        offline: bool,
        device: str,
        seed: int,
        grain: bool,
        style_strength: float,
        request: gr.Request,
        progress=progress,
    ) -> Iterator[tuple]:
        try:
            with controller.claim(request.session_hash or "local"):
                yield (
                    None,
                    None,
                    None,
                    "",
                    None,
                    "**Working.** Validating this request…",
                    gr.Button(interactive=False),
                )
                parameters = UiParameters(
                    content,
                    mode,
                    preset,
                    reference,
                    intensity,
                    quality,
                    preservation,
                    custom_steps,
                    steps,
                    limit_output,
                    max_size,
                    offline,
                    device,
                    seed,
                    grain,
                    style_strength,
                )
                rendered = controller.generate(
                    parameters, stage=lambda text: progress(None, desc=text)
                )
                yield (
                    rendered.image,
                    rendered.comparison,
                    str(rendered.download),
                    rendered.summary,
                    rendered.metadata,
                    "**Complete.** Compare and download your result.",
                    gr.Button(interactive=True),
                )
        except Exception as error:
            logging.getLogger(__name__).error(
                "UI request stopped (%s)", type(error).__name__
            )
            yield (
                None,
                None,
                None,
                "",
                None,
                f"**Could not generate:** {friendly_error(error)}",
                gr.Button(interactive=True),
            )

    return run
