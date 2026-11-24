"""Training evidence and isolated experimental inference interface."""

import logging

import gradio as gr

from moodshift.ui.controller import UiController
from moodshift.ui.experiments import (
    RUN_NAME,
    ExperimentError,
    experiment_info,
    infer_with_reconstruction,
)


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
        buttons=["download", "fullscreen"],
        visible=graph is not None,
    )
    gr.Markdown("### Held-out validation examples")
    gr.Markdown(
        "The decoder itself outputs 96×96. Each displayed high-resolution result "
        "reconstructs its recorded color changes on the matching original crop "
        "with a 1.5× visibility adjustment. Missing originals are not substituted."
    )
    if samples:
        gr.Gallery(
            value=samples,
            columns=3,
            label="Content, style, and trained output",
            height="auto",
            object_fit="contain",
            interactive=False,
            buttons=["download", "download_all", "fullscreen"],
        )
    else:
        gr.Markdown("Validation example images are not available locally.")
    gr.Markdown("""### Experimental Trained Decoder
This separate CPU model produces subtle style changes and is not equivalent in visual
quality to the Studio's optimization-based neural transfer. Inputs are center-cropped
and resized to 96 × 96. The decoder itself generates 96x96 images. The high-resolution
version preserves original image detail through postprocessing. Both outputs use the
same centered content crop; no grade or optimization engine is substituted.""")
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
    amplification = gr.Slider(
        1.0,
        2.0,
        value=1.5,
        step=0.05,
        label="Decoder color-change amplification",
        info="Amplifies only the decoder-predicted changes, with safety limits.",
    )
    button = gr.Button(
        "Generate experimental result", variant="primary", interactive=available
    )
    status = gr.Markdown(
        "Ready for CPU inference."
        if available
        else "Experimental inference is unavailable. Check the evidence above."
    )
    gr.Markdown(
        "**Detail-preserving reconstruction (decoder color change, amplified "
        "for visibility).**"
    )
    with gr.Row():
        high_result = gr.Image(
            label="Detail-preserving reconstruction (recommended)",
            interactive=False,
            format="png",
            height=384,
            buttons=["download", "fullscreen"],
        )
        comparison = gr.ImageSlider(
            label="Original content crop / detail-preserving reconstruction",
            interactive=False,
            type="pil",
            format="png",
            height=384,
            elem_id="experimental-comparison",
        )
    high_download = gr.DownloadButton("Download reconstructed PNG", value=None)
    with gr.Accordion("Native 96 × 96 trained decoder output", open=False):
        result = gr.Image(
            label="Experimental output",
            interactive=False,
            format="png",
            height=384,
            width=384,
            elem_id="experimental-result",
            buttons=["download", "fullscreen"],
        )
        download = gr.DownloadButton("Download native 96 × 96 PNG", value=None)

    def generate(
        content_path, style_path, selected, amplification, request: gr.Request
    ):
        try:
            with controller.claim(request.session_hash or "local"):
                with controller._processing_lock:
                    controller._prune_exports()
                    image, pair, file, native, native_file = infer_with_reconstruction(
                        content_path,
                        style_path,
                        selected,
                        controller.output_dir,
                        amplification=amplification,
                    )
                return (
                    image,
                    pair,
                    file,
                    native,
                    native_file,
                    f"**Complete.** Reconstruction: {image.width} × {image.height}; "
                    "native decoder output: 96 × 96 RGB; "
                    f"color change {amplification:g}×.",
                )
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
            return None, None, None, None, None, f"**Could not generate:** {message}"

    button.click(
        generate,
        [content, style, checkpoint, amplification],
        [high_result, comparison, high_download, result, download, status],
        concurrency_limit=1,
        concurrency_id="moodshift-generation",
        trigger_mode="once",
        api_name="generate_experimental",
    )
