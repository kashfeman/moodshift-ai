"""Gradio Blocks entrypoint for the existing mood-transfer service."""

import argparse
import atexit

import gradio as gr

from moodshift.services.models import MoodSettings
from moodshift.ui.callbacks import make_generate
from moodshift.ui.components import (
    CSS,
    HERO,
    app_theme,
    build_form,
    project_information,
)
from moodshift.ui.controller import UiController
from moodshift.ui.helpers import friendly_error, mode_presentation


def create_app(controller: UiController | None = None) -> gr.Blocks:
    """Build local UI/queue without launching a server or initializing a model."""
    controller = controller or UiController()
    atexit.register(controller.close)
    with gr.Blocks(
        title="MoodShift AI",
        analytics_enabled=False,
        delete_cache=(600, 3600),
        fill_width=True,
    ) as app:
        gr.HTML(HERO)
        with gr.Row(elem_id="workspace"):
            with gr.Column(scale=4, min_width=320, elem_id="controls"):
                form = build_form()
                generate = gr.Button(
                    "Generate →", variant="primary", elem_id="generate"
                )
            with gr.Column(scale=6, min_width=360, elem_id="results"):
                gr.Markdown("### 03 · A new atmosphere")
                status = gr.Markdown(
                    "**Ready.** Upload a photograph to begin.", elem_id="status"
                )
                with gr.Tab("Before / after"):
                    comparison = gr.ImageSlider(
                        label="Before ← drag → After",
                        interactive=False,
                        type="pil",
                        format="png",
                        height=360,
                    )
                with gr.Tab("Result"):
                    result = gr.Image(
                        label="Generated RGB image",
                        interactive=False,
                        height=360,
                        format="png",
                        buttons=["fullscreen"],
                    )
                download = gr.DownloadButton(
                    "Download lossless PNG", value=None, variant="secondary"
                )
                summary = gr.Markdown(
                    "Execution facts will appear here after generation."
                )
                with gr.Accordion("Effective settings & metadata", open=False):
                    metadata = gr.JSON(label="Actual processing metadata")
        gr.Examples(
            examples=[
                ["grade", "cinematic", 70, "preview"],
                ["grade", "sunset", 65, "preview"],
                ["hybrid", "dreamy", 50, "preview"],
            ],
            inputs=[form.mode, form.preset, form.intensity, form.quality],
            label="Suggested settings · upload your own image",
            cache_examples=False,
        )
        project_information()

        def preview(path: str | None) -> tuple:
            try:
                image = controller.preview(path)
                message = (
                    f"Validated · {image.width} × {image.height} RGB"
                    if image
                    else "Upload PNG, JPEG or WebP · up to 10 MiB."
                )
                return image, message
            except Exception as error:
                return None, f"**Upload needs attention:** {friendly_error(error)}"

        def presentation(
            mode: str, reference: str | None, intensity: float, style_strength: float
        ) -> tuple:
            view = mode_presentation(mode, bool(reference), intensity, style_strength)
            return (
                gr.Column(visible=view.neural_visible),
                view.explanation,
                gr.Dropdown(interactive=view.preset_enabled),
                gr.Checkbox(interactive=view.grain_enabled),
                gr.File(value=None) if mode == "grade" else gr.skip(),
            )

        def quality_defaults(quality: str) -> tuple[int, int]:
            defaults = MoodSettings(quality=quality).neural_settings()
            return defaults.steps, defaults.max_size

        form.content.change(
            preview,
            form.content,
            [form.preview, form.upload_status],
            queue=False,
            api_visibility="private",
        )
        gr.on(
            [
                form.mode.change,
                form.reference.change,
                form.intensity.change,
                form.style_strength.change,
            ],
            presentation,
            [form.mode, form.reference, form.intensity, form.style_strength],
            [
                form.neural_group,
                form.explanation,
                form.preset,
                form.grain,
                form.reference,
            ],
            queue=False,
            api_visibility="private",
        )
        form.quality.change(
            quality_defaults,
            form.quality,
            [form.steps, form.max_size],
            queue=False,
            api_visibility="private",
        )
        form.custom_steps.change(
            lambda checked: gr.Slider(visible=checked),
            form.custom_steps,
            form.steps,
            queue=False,
            api_visibility="private",
        )
        form.limit_output.change(
            lambda checked: gr.Slider(visible=checked),
            form.limit_output,
            form.max_size,
            queue=False,
            api_visibility="private",
        )
        run = make_generate(controller)

        generate.click(
            run,
            form.inputs(),
            [result, comparison, download, summary, metadata, status, generate],
            concurrency_limit=1,
            concurrency_id="moodshift-generation",
            trigger_mode="once",
            show_progress="full",
            api_name="generate",
        )
    app.queue(max_size=8, default_concurrency_limit=1)
    return app


def main(argv: list[str] | None = None) -> int:
    """Launch locally by default; sharing/network binding require explicit flags."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    controller = UiController()
    try:
        create_app(controller).launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share,
            inbrowser=False,
            show_error=False,
            max_file_size=controller.config.max_upload_bytes,
            allowed_paths=[str(controller.output_dir)],
            theme=app_theme(),
            css=CSS,
        )
    finally:
        controller.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
