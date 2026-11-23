"""Training evidence and isolated experimental inference interface."""

import logging

import gradio as gr

from moodshift.ui.controller import UiController
from moodshift.ui.experiments import RUN_NAME, ExperimentError, experiment_info, infer


def build_experiments(controller: UiController) -> None:
    """Show local evidence and expose a separate, pinned-checkpoint inference flow."""
    info, rows, graph, samples, available = experiment_info()
    gr.Markdown("## Training & Experiments")
    gr.Markdown(info)
    gr.Markdown("### Recorded losses")
    gr.Dataframe(
        headers=["Epoch", "Training loss", "Validation loss"],
        value=rows,
        interactive=False,
        label="Checkpoint training history",
    )
    gr.Image(
        value=graph,
        label="Recorded training and validation loss curve",
        interactive=False,
        format="png",
        visible=graph is not None,
    )
    gr.Markdown("### Held-out validation examples")
    if samples:
        gr.Gallery(
            value=samples,
            columns=3,
            label="Content, style, and trained output",
            height="auto",
            object_fit="contain",
            interactive=False,
        )
    else:
        gr.Markdown("Validation example images are not available locally.")
    gr.Markdown("""### Experimental Trained Decoder
This separate CPU model produces subtle style changes and is not equivalent in visual
quality to the Studio's optimization-based neural transfer. Inputs are center-cropped
and resized to 96 × 96. No grade or optimization engine is substituted if it fails.""")
    content = gr.File(
        label="Content image",
        type="filepath",
        file_types=[".png", ".jpg", ".jpeg", ".webp"],
    )
    style = gr.File(
        label="Style reference",
        type="filepath",
        file_types=[".png", ".jpg", ".jpeg", ".webp"],
    )
    checkpoint = gr.Dropdown(
        choices=[("Guided decoder · epoch 5", RUN_NAME)] if available else [],
        value=RUN_NAME if available else None,
        label="Trusted local checkpoint",
        allow_custom_value=False,
    )
    button = gr.Button(
        "Generate experimental result", variant="primary", interactive=available
    )
    status = gr.Markdown(
        "Ready for CPU inference."
        if available
        else "Experimental inference is unavailable. Check the evidence above."
    )
    with gr.Row():
        result = gr.Image(label="Experimental output", interactive=False, format="png")
        comparison = gr.ImageSlider(
            label="Cropped content / trained output",
            interactive=False,
            type="pil",
            format="png",
        )
    download = gr.DownloadButton("Download experimental PNG", value=None)

    def generate(content_path, style_path, selected, request: gr.Request):
        try:
            with controller.claim(request.session_hash or "local"):
                with controller._processing_lock:
                    controller._prune_exports()
                    image, pair, file = infer(
                        content_path, style_path, selected, controller.output_dir
                    )
                return image, pair, file, "**Complete.** Output is 96 × 96 RGB."
        except Exception as error:
            logging.getLogger(__name__).error(
                "Experimental inference stopped (%s)", type(error).__name__
            )
            if isinstance(error, ExperimentError):
                message = str(error)
            else:
                message = (
                    "Processing failed. Check image validity and cached "
                    "official weights."
                )
            return None, None, None, f"**Could not generate:** {message}"

    button.click(
        generate,
        [content, style, checkpoint],
        [result, comparison, download, status],
        concurrency_limit=1,
        concurrency_id="moodshift-generation",
        trigger_mode="once",
        api_name="generate_experimental",
    )
