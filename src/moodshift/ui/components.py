"""Reusable Gradio form, result, and project-information sections."""

from dataclasses import dataclass

import gradio as gr

from moodshift.presets import PRESETS
from moodshift.ui.helpers import mode_presentation

CSS = """
[class*="gradio-container"] {max-width: 1440px !important; width: 100% !important;
  min-width: 0 !important; margin: auto; box-sizing: border-box; padding: 16px;}
#hero {padding: 22px 28px; border-radius: 22px; margin-bottom: 18px;
  font-family: system-ui, sans-serif;
  background: linear-gradient(120deg, #102e35, #1d2444); color: #eefbf8;}
#hero h1 {color: #eefbf8; font-size: clamp(2rem, 4vw, 2.65rem); letter-spacing: -.04em;
  margin: 8px 0 10px; line-height: 1.05;}
#hero p {color: #c1d9d9; max-width: 710px; margin: 0; line-height: 1.6;}
#hero .eyebrow {color: #78e1c5; letter-spacing: .16em; font-size: 11px;
  font-weight: 700; text-transform: uppercase;}
#workspace {gap: 24px; align-items: start; min-width: 0; width: 100%;}
#controls, #results {border: 1px solid var(--border-color-primary);
  border-radius: 18px; padding: 20px; background: var(--block-background-fill);}
#generate {min-height: 48px; font-weight: 650; letter-spacing: .02em;}
#results {position: sticky; top: 16px;}
.section-label {font-size: 12px; text-transform: uppercase; letter-spacing: .12em;}
#status {border-radius: 12px; padding: 12px 16px;
  background: var(--background-fill-secondary); min-height: 44px;}
@media (max-width: 760px) {
  #workspace {flex-direction: column !important;}
  #controls, #results {min-width: 0 !important; width: 100% !important;
    box-sizing: border-box;}
  #hero {padding: 22px;} #controls, #results {padding: 14px;}
  #results {position: static;}
}
"""

HERO = """<div id="hero"><div class="eyebrow">
Digital Image Processing · University project</div>
<h1>MoodShift AI</h1><p>Give your photographs a different atmosphere.
Choose a precise color grade, genuine neural reference transfer,
or combine both.</p></div>"""


@dataclass(slots=True)
class Form:
    """Named form components in callback input order."""

    content: gr.File
    mode: gr.Radio
    preset: gr.Dropdown
    reference: gr.File
    intensity: gr.Slider
    quality: gr.Dropdown
    preservation: gr.Slider
    custom_steps: gr.Checkbox
    steps: gr.Slider
    limit_output: gr.Checkbox
    max_size: gr.Slider
    offline: gr.Checkbox
    device: gr.Dropdown
    seed: gr.Number
    grain: gr.Checkbox
    style_strength: gr.Slider
    neural_group: gr.Column
    explanation: gr.Markdown
    preview: gr.Image
    upload_status: gr.Markdown

    def inputs(self) -> list:
        """Return the explicit order consumed by the generate callback."""
        return [
            self.content,
            self.mode,
            self.preset,
            self.reference,
            self.intensity,
            self.quality,
            self.preservation,
            self.custom_steps,
            self.steps,
            self.limit_output,
            self.max_size,
            self.offline,
            self.device,
            self.seed,
            self.grain,
            self.style_strength,
        ]


def build_form() -> Form:
    """Build a compact, mode-aware form with a fast non-neural initial state."""
    gr.Markdown("### 01 · Your image")
    content = gr.File(
        label="Content image",
        type="filepath",
        file_types=[".png", ".jpg", ".jpeg", ".webp"],
        height=130,
    )
    upload_status = gr.Markdown("PNG, JPEG or WebP · up to 10 MiB · normalized to RGB")
    with gr.Accordion("Validated content preview", open=False):
        preview = gr.Image(
            label="Normalized RGB",
            interactive=False,
            height=150,
            buttons=["fullscreen"],
            format="png",
            show_label=False,
        )
    gr.Markdown("### 02 · Set the atmosphere")
    mode = gr.Radio(
        choices=[
            ("Color grade", "grade"),
            ("Neural transfer", "neural"),
            ("Hybrid", "hybrid"),
        ],
        value="grade",
        label="Processing mode",
    )
    preset = gr.Dropdown(
        choices=[(p.name, p.id) for p in PRESETS],
        value="cinematic",
        label="Mood preset",
        allow_custom_value=False,
    )
    explanation = gr.Markdown(mode_presentation("grade", False, 70).explanation)
    intensity = gr.Slider(
        0,
        100,
        value=70,
        step=1,
        label="Preset / result intensity",
        info="Overall preset intensity. Zero returns unchanged normalized content.",
    )
    with gr.Column(visible=False) as neural_group:
        style_strength = gr.Slider(
            0,
            1,
            value=1.0,
            step=0.05,
            label="Neural style strength",
            info="Weights neural style loss. Zero skips inference and references.",
        )
        reference = gr.File(
            label="Optional style reference",
            type="filepath",
            file_types=[".png", ".jpg", ".jpeg", ".webp"],
        )
        quality = gr.Dropdown(
            choices=[
                ("Preview · 20 steps / 128px", "preview"),
                ("Balanced · 60 steps / 256px", "balanced"),
                ("High · 120 steps / 384px", "high"),
            ],
            value="preview",
            label="Neural quality",
            allow_custom_value=False,
        )
        preservation = gr.Slider(
            0,
            1,
            value=0.7,
            step=0.05,
            label="Content preservation",
            info="Weights spatial content loss; higher favors the scene structure.",
        )
        with gr.Accordion("Neural settings", open=False):
            offline = gr.Checkbox(
                value=True,
                label="Offline · require cached model weights",
                info="Turn off to allow the single official first-run model download.",
            )
            device = gr.Dropdown(
                choices=[
                    ("Auto (CUDA if available)", "auto"),
                    ("CPU", "cpu"),
                    ("CUDA", "cuda"),
                ],
                value="auto",
                label="Compute device",
            )
            custom_steps = gr.Checkbox(
                value=False, label="Customize optimization steps"
            )
            steps = gr.Slider(
                1, 500, value=20, step=1, visible=False, label="Optimization steps"
            )
    with gr.Accordion("Output & reproducibility", open=False):
        limit_output = gr.Checkbox(
            value=False,
            label="Limit output resolution",
            info="Off preserves original dimensions. Quality sets neural resolution.",
        )
        max_size = gr.Slider(
            32,
            512,
            value=128,
            step=1,
            visible=False,
            label="Maximum output / processing side (px)",
        )
        seed = gr.Number(
            value=0,
            precision=0,
            minimum=0,
            maximum=2**32 - 1,
            label="Seed",
            info="Used for neural initialization and optional Vintage grain.",
        )
        grain = gr.Checkbox(value=True, label="Allow Vintage film grain")
    return Form(
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
        neural_group,
        explanation,
        preview,
        upload_status,
    )


def app_theme() -> gr.themes.Soft:
    """Use system fonts and quiet labels with native light/dark theme variables."""
    return gr.themes.Soft(
        primary_hue="teal", secondary_hue="slate", font=["system-ui", "sans-serif"]
    ).set(
        block_label_background_fill="transparent",
        block_label_background_fill_dark="transparent",
        block_label_text_color="*neutral_600",
        block_label_text_color_dark="*neutral_300",
        block_label_border_width="0px",
        block_label_shadow="none",
    )


def project_information() -> None:
    """Explain a complete workflow, actual model behavior, and honest tradeoffs."""
    with gr.Accordion("How to Use", open=False):
        gr.Markdown("""1. Upload a content image; its preview appears after validation.
2. Choose **Color grade**, **Neural transfer**, or **Hybrid**.
3. Select a mood preset.
4. Optionally upload a style image in a neural mode. Without one, the selected
preset supplies a clearly labeled **synthetic** reference.
5. Adjust style intensity, quality, and optional advanced controls.
6. Click **Generate** once and wait for processing to finish.
7. Compare the images and download the lossless PNG.

Result intensity blends the final appearance with the original; zero skips all
effects. Neural style strength weights the CNN style loss independently. At zero,
Neural returns unchanged content and Hybrid runs only its photographic grade.""")
    with gr.Accordion("Project Info", open=False):
        gr.Markdown(
            "MoodShift AI is a university Digital Image Processing project. "
            "The interface calls the existing typed MoodTransferService. "
            "Photographic grading uses CIELAB, smooth tonal masks, tone curves, "
            "controlled bloom and grain. Neural modes use real pretrained features."
        )
    with gr.Accordion("Limitations", open=False):
        gr.Markdown("""Neural optimization can be slow on CPU and can introduce texture
artifacts. Preview uses fewer steps and pixels; High costs more time and memory.
No fixed runtime or aesthetic quality is promised. Restoring original output
dimensions cannot restore fine detail lost at the internal neural resolution.
Images are processed as 8-bit RGB; ICC/HDR workflows are not supported.
Only one generation runs at a time. Downloads are temporary; save your result.
Final submission polish and broader integration review remain a later milestone.""")
    with gr.Accordion("Technical Details", open=False):
        gr.Markdown("""**Model:** SqueezeNet 1.1, official ImageNet1K_V1 weights.
Frozen intermediate features guide Adam optimization using content MSE,
normalized Gram style loss, and total variation.

**Grade:** photographic DIP only, no neural inference or reference application.
**Neural:** real reference-guided optimization without a post-grade.
**Hybrid:** real neural optimization followed by the selected photographic grade.

External references replace built-in references. In Neural mode the preset is
disabled while an external reference is supplied. In Hybrid it still supplies the
post-grade. The selected device is resolved on execution; the result reports the
actual backend, device, reference source and settings. No paid API or key is used.""")
