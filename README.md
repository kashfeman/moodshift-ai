# MoodShift AI

A university Digital Image Processing project for exploring mood-based image
styling. **Studio and the experimental trained-decoder dashboard are implemented.**

Implemented:

- Installable, typed Python package using a `src` layout.
- Immutable configuration with validated limits and default settings.
- Immutable metadata and typed grading recipes for Cinematic, Sunset, Cold Blue,
  Vintage, and Dreamy.
- Bounded PNG/JPEG/WebP loading, EXIF orientation, and RGB normalization.
- Aspect-preserving resize, RGB/LAB conversion, and image statistics.
- PNG/JPEG export without overwriting existing files.
- Optimization-based neural transfer with frozen pretrained SqueezeNet features,
  content loss, normalized Gram style loss, total variation, and Adam.
- CPU/CUDA selection, bounded quality controls, and a local command-line demo.
- Real CIELAB photographic grading, deterministic synthetic style references,
  and a service with grade-only, neural-only, and hybrid modes.
- Generated-image tests, Ruff checks, and GitHub Actions for Python 3.12.
- Responsive Gradio Blocks UI with mode-aware controls, validated uploads,
  before/after slider, PNG download, and actual processing metadata.
- Content-guided Hybrid reconstruction that preserves source detail while applying
  the genuine neural color residual; direct artistic Neural output remains separate.

NOT YET IMPLEMENTED:

- Automatic mood recognition/reference selection.
- Production deployment and broad photographic/user-study evaluation.

Final submission materials: [project report](docs/submission/PROJECT_REPORT.md),
[5–7 minute demo](docs/submission/DEMO_SCRIPT.md),
[actual QA results](docs/submission/TEST_SUMMARY.md), and
[submission checklist](docs/submission/SUBMISSION_CHECKLIST.md).
The report records a genuine before/after investigation and a controlled 20/60-step,
128/256/384-pixel comparison. The revised distributable ZIP excludes photographic
evidence pending permission review. Hybrid visibly reduced artifacts on the investigated
landscape; more steps did not consistently improve its appearance. Neural-only
output can still introduce texture artifacts. No broad aesthetic guarantee is made.

Named presets now perform actual photographic grading. Grade processing uses only
NumPy/OpenCV/Pillow and does not run the neural engine. The dashboard may load
the cached encoder at application startup to check experimental availability. Neural modes optimize real pretrained
features against a user reference or a clearly labeled synthetic preset reference.
No paid API is used. Only neural modes may download the official checkpoint when
needed and allowed; grade and zero-intensity processing do not invoke the neural model. The dashboard
may check cached model availability when the app starts.

## Launch the Gradio app

From the repository root, using the existing Python 3.12 environment:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m moodshift.ui.app
```

Open **http://127.0.0.1:7860**. The default is a local-only server with no public
share link. `--host` and `--port` change the binding; `--share` explicitly enables
Gradio's public tunnel. Stop with Ctrl+C. No separate JS/React frontend or custom
FastAPI application is built; Gradio supplies its own web-server dependencies.

The UI uses Gradio >=6.32,<7 (verified with 6.32.0); theme/CSS are supplied at
launch as required by [Gradio 6](https://gradio.app/guides/gradio-6-migration-guide).
There are no external font downloads or paid APIs.

**Workflow:** upload content, choose Grade/Neural/Hybrid, select a preset, optionally
upload a neural reference, adjust controls, Generate, compare, and download PNG.
Uploads stay as encoded files until the existing loader validates PNG/JPEG/WebP,
10 MiB size, dimensions/pixels, and EXIF/RGB normalization. Neural controls hide
in Grade mode; no reference is applied there. External references disable the
preset in Neural mode, and override only the built-in reference in Hybrid mode.
Without an external reference, neural modes clearly identify the selected
preset's deterministic **synthetic** built-in reference.

UI controls include independent **Preset/result intensity** (0..100) and **Neural
style strength** (0..1), content preservation, Preview/Balanced/High quality,
optional custom optimization steps, optional output size limit, device, offline
mode, seed, and Vintage grain. Defaults favor quick Grade mode and Preview neural
quality. Selecting quality resets displayed step/resolution defaults; custom steps
are used only when enabled. An unchecked output limit preserves original dimensions.

- Result intensity zero returns exact normalized source pixels with no effects.
- Neural strength zero skips neural inference/reference application: Neural mode
  returns unchanged content; Hybrid still applies its photographic grade.
- Offline is on by default. Turn it off only to allow the existing loader's
  official first-run checkpoint download; no download occurs just by opening the UI.

For weights cached by the previous milestones in this repository, set the normal
PyTorch cache environment variable **before** starting the app:

```powershell
$env:TORCH_HOME = Join-Path (Get-Location) '.cache\torch'
python -m moodshift.ui.app --port 7860
```

Generation is serialized with a bounded queue. The button disables while running,
duplicate session submissions are guarded, and progress describes real stages;
there is no invented per-step percentage or fixed runtime. Errors clear stale
outputs, explain the issue, and restore the button. Actual backend/device and
effective controls come from returned service metadata, not from UI selections.

The before/after slider uses matched dimensions and lossless PNG previews. The
download is exported via the existing non-overwriting RGB exporter to a unique
owned temporary path; source images are never overwritten. Gradio cache files
expire after an hour (checked every ten minutes). Source export files are expired
on the next generation or removed on app shutdown. Downloads should be saved
promptly; these are temporary demo artifacts, not persistent image storage.

Suggested-setting examples require your own image and do not reference absent
sample files. How to Use, Project Info, Limitations, and Technical Details accordions
explain the project and its CPU/quality tradeoffs. Final polish, broader photographic
evaluation, and final submission integration review remain later work.

## Development

Use Python >=3.12,<3.13 (workstation: 3.12.10). Preserve the existing virtual
environment; do not reinstall Python or switch global interpreters. From the root:

```powershell
.venv\Scripts\Activate.ps1
python --version
python -m pip install --no-cache-dir torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
python -m pip check
git diff --check
```

For a fresh checkout only, create `.venv` with Python 3.12 using
`python -m venv .venv`. On POSIX, activate with `source .venv/bin/activate`.
Runtime dependencies are Pillow >=11.1,<13, NumPy >=2.2,<3, and headless OpenCV
>=4.11,<5, plus PyTorch >=2.10,<2.11 and matching TorchVision >=0.25,<0.26.
The CPU-index install above avoids GPU libraries on CPU workstations; CI uses it
too. These versions follow the [official compatible package pair](https://pytorch.org/get-started/previous-versions/).
Pytest and Ruff remain developer dependencies; Gradio supplies the frontend. CUDA use
requires a compatible CUDA PyTorch installation and actual supported hardware;
the engine never installs or switches package builds itself.

## Configuration and presets

`moodshift.config.AppConfig` defaults to a 10 MiB encoded upload limit, maximum
width and height of 4096 pixels, and 16 million decoded pixels. All limits are
inclusive and enforced together by image loading. The default
preset ID is `cinematic`, strength is `0.5` (allowed range `[0, 1]`), and output
format is `PNG` (`JPEG` is also accepted). Configuration validates setting values;
it does not inspect uploads. Custom preset IDs are accepted by configuration;
services will resolve them against the catalog using `get_preset`.

`moodshift.presets.PRESETS` contains the five immutable metadata/recipe entries.
`get_preset(id)` raises `KeyError` for unknown IDs.

See [the architecture plan](docs/ARCHITECTURE.md) for module boundaries. Store
future tracked preset assets in `assets/presets/`; keep generated images in the
ignored `outputs/` directory and downloaded weights in ignored `weights/`.

## Image processing

```python
from moodshift.imaging.color import pil_to_rgb_array, rgb_to_lab
from moodshift.imaging.io import export_image, load_image
from moodshift.imaging.statistics import image_statistics
from moodshift.imaging.transforms import resize_to_fit

image = load_image("input.png")
preview = resize_to_fit(image, (1024, 1024))
rgb = pil_to_rgb_array(preview)
lab = rgb_to_lab(rgb)
stats = image_statistics(rgb)
export_image(preview, "preview.png")  # destination must not exist
```

Loading detects format from content, checks encoded bytes and raw dimensions
before decoding, and rejects animations and Pillow bomb warnings. Oriented
dimensions are checked before rotation; PNG EXIF access can trigger decoding.
Typed errors live in `moodshift.imaging.exceptions`. Unknown formats and corrupt
bytes cannot always be distinguished by the decoder. Grayscale becomes RGB;
transparency composites over white. High-bit-depth/float modes are rejected.
ICC profiles are not applied; CMYK uses basic Pillow conversion. Metadata is
stripped. These are 8-bit processing utilities, not color-managed archival tools.

Resizing uses LANCZOS without stretching or upscaling (at most one pixel rounding).
Arrays are independent H x W x 3 uint8 RGB, never implicitly BGR or scaled floats.
LAB uses float32 CIE units (L 0..100, signed a/b, D65). Reverse conversion rounds
and clips out-of-gamut colors. Statistics measure channel means and population
deviations, plus mean/std of gamma-encoded BT.709 luma in 0..1.

Export strips metadata, supports lossless PNG and lossy JPEG (quality 95 default,
no chroma subsampling), and requires an existing parent directory. Encoding
finishes before exclusive file creation; existing paths are never overwritten.
A failed disk write may leave a partial new file and raises `ImageExportError`.
Callers choose local paths; these utilities are not a server path sandbox or an
isolated decoder process. Source images and files are not modified.

If Pytest reports cache `WinError 5`, inspect `.pytest_cache` ownership under the
same test user. Do not use administrator access or alter system ACLs. The warning
did not reproduce during inspection. Safe fallback commands are
`python -m pytest -q -o cache_dir=.pytest_cache_local` (ignored by Git), or
`python -m pytest -q -p no:cacheprovider` if caching is unnecessary.

## Mood presets and service CLI

```powershell
python -m moodshift.services.demo --content "content.png" --preset cinematic --mode grade --intensity 75 --output "cinematic.png"
python -m moodshift.services.demo --content "content.png" --preset dreamy --mode hybrid --quality preview --intensity 60 --offline --output "dreamy-hybrid.png"
python -m moodshift.services.demo --content "content.png" --reference "reference.jpg" --mode neural --quality balanced --intensity 65 --output "reference-transfer.png"
```

| Mode | Processing | Selection |
| --- | --- | --- |
| `grade` | CIELAB color/tonal grading, bloom/grain where configured; no neural model | Preset required; references rejected |
| `neural` | Genuine SqueezeNet feature optimization; no post-grading | User reference **or** preset synthetic reference |
| `hybrid` | Genuine neural optimization followed by named preset grading | Preset required; optional user reference overrides the synthetic reference |

Combining a preset and user reference in `neural` mode is rejected to avoid
silently ignoring the preset; select `hybrid` for that combination. Existing files
are never overwritten. CLI output reports actual backend, device, reference
source, executed neural steps, and output dimensions. Errors do not produce a
substitute image or falsely report completed inference.

| Preset | Photographic recipe |
| --- | --- |
| Cinematic | Teal a/b offsets in shadows, warm highlights, deeper continuous contrast, modest saturation |
| Sunset | Golden highlights, orange/pink midtones, lifted gamma, restrained vibrance |
| Cold Blue | Blue/cyan shadows and midtones, reduced warmth/saturation, controlled depth |
| Vintage | Warm faded chroma, lifted blacks, compressed whites, reduced saturation, delicate seeded monochrome grain |
| Dreamy | Pastel chroma, gentler contrast, brighter gamma, highlight-only soft bloom |

Masks form a smooth shadow/midtone/highlight partition of CIELAB L. Chroma-only
operations leave L unchanged before tone adjustment and RGB gamut clipping.
Offsets taper near black and white. Vibrance favors muted colors and reduces its
boost on warm hues; this is a hue heuristic, not face or skin detection. Tone
curves are monotonic and preserve smooth ramps. Bloom softly screens blurred
highlights without blurring the whole scene. Grain uses a local seeded generator
and a shared RGB-channel noise field. Use `--no-grain` to disable Vintage's final
grain pass; neural texture/artifacts are not removed by that option.

Intensity is a **single final interpolation in encoded RGB**, from normalized
original pixels to the full processed appearance. Zero returns exact full-size
normalized pixels without model loading, grading, or synthetic-reference creation.
All supplied input files and request combinations are still validated. Export as
PNG for exact pixels; JPEG encoding remains lossy. Neural modes default to style
strength 1 and use intensity for the final blend. Milestone 5 adds an optional
`MoodSettings.neural_style_strength` pass-through for the UI (default unchanged),
independent of intensity. Zero skips inference with truthful metadata. Content
preservation weights the engine's content loss; existing CLI defaults are unchanged.

### Service quality and dimensions

| Quality | Neural Adam steps | Internal maximum side |
| --- | --- | --- |
| `preview` | 20 | 128px |
| `balanced` (default) | 60 | 256px |
| `high` | 120 | 384px |

These are the **service** quality defaults; the lower-level Milestone 3 engine
defaults below remain unchanged. Override `--steps` (1..500) or `--max-size`
(32..512). Grade-only requests execute zero neural steps regardless of quality.
Without explicit `--max-size`, grade-only processes at the original size. Neural
pixels are optimized at the quality resolution and LANCZOS-resampled back onto
the original grid, preserving dimensions and scene layout without cropping.
Restoring dimensions does not restore lost fine detail. An explicit `--max-size`
also limits the final output without upscaling the source. Each neural input axis
must remain at least 17px after processing resize. Zero ignores output size limits
so that the exact-original guarantee holds.

High mode costs more CPU time/memory; no fixed runtime is promised. The service
does not change global thread counts; the demo CLI uses two CPU threads for neural
requests. Full-size grades allocate float image buffers; explicitly limit large
images if memory is tight. CUDA is supported by the existing engine but untested
on this CPU-only workstation. Low-step neural previews can have conspicuous
texture artifacts. The synthetic references are simple palette/texture studies,
not photographs or specialized trained mood models; grade-only is often the
more restrained photographic result. ICC/HDR limitations from Milestone 2 remain.

```python
from moodshift.services.models import MoodSettings
from moodshift.services.mood_transfer import MoodTransferService

service = MoodTransferService(allow_download=False)
result = service.transfer(
    "content.png", preset_id="sunset", settings=MoodSettings(mode="grade", intensity=80)
)
image, metadata = result.image, result.metadata
```

Result metadata is frozen and reports preset ID, requested mode, actual backend,
device, whether neural/grading stages ran, applied reference source, original,
neural and final sizes, and effective settings. Zero reports engine `none` and
reference `none`; grade reports `numpy-opencv`, CPU, zero neural steps. Injected
unit-test backends are explicitly labeled `injected-neural-backend`, never
pretrained SqueezeNet. Reuse a service instance for sequential neural calls.

### Synthetic references and reproducible examples

Built-in references are deterministic 128x128 mathematical color/texture patterns
graded using the same preset registry. They are generated locally, temporarily
encoded for the existing path-based engine, then cleaned up. They are **synthetic
style inputs**, not AI outputs or third-party photographs. Reference grain uses
fixed seed 0; `--seed` controls neural initialization and the final grain pass.
See [asset provenance](assets/presets/README.md).

Generate a new set of demo fixtures and before/after examples:

```powershell
python -m moodshift.services.examples --output-dir "outputs/my-comparison"
$env:TORCH_HOME = Join-Path (Get-Location) '.cache\torch'
python -m moodshift.services.examples --output-dir "outputs/my-hybrid-comparison" --hybrid
```

The directory must be new. The optional hybrid uses cached weights only. The
synthetic landscape, five real grade outputs, contact sheet, result metadata, and
optional hybrid output remain local and ignored by Git under `outputs/`.
Milestone 4 examples were generated in `outputs/milestone4/`, reloaded and visually
inspected: mean absolute changes were 5.059 (Cinematic), 9.176 (Sunset), 6.453
(Cold Blue), 13.805 (Vintage), and 12.217 (Dreamy) on the uint8 scale. A real
20-step Cinematic hybrid also completed on CPU. These tests demonstrate working
operations and distinct looks, not a benchmark of photographic quality.

## Lower-level neural transfer and CLI (Milestone 3)

```powershell
python -m moodshift.engine.demo --content "content.png" --style "reference.jpg" --output "result.png" --quality preview --strength 0.6 --content-preservation 0.7 --steps 60 --max-size 192 --device auto
```

Parents must exist and the output path must be new. The CLI prints the actual
device and completion size, without printing input paths. Inputs use the existing
10 MiB/dimension/pixel validation. Both images retain their own aspect ratios,
never upscale, and must remain at least 17 pixels per axis after downsampling.
The result has the downsampled content dimensions. Style strength 0 returns the
full-size decoded RGB content without resizing or loading weights; export to PNG
to retain those exact pixels. Both input files are still validated.

| Control | Default and bounds |
| --- | --- |
| `--strength` | 0.5; 0..1, weights neural style loss |
| `--content-preservation` | 0.5; 0..1, weights spatial content loss |
| `--quality preview` | 60 Adam steps, longest side at most 192px |
| `--quality high` | 180 steps, longest side at most 384px |
| `--steps` / `--max-size` | Override mode defaults; 1..500 steps, 32..512px |
| `--seed` | 0; 0..4294967295, local initialization noise generator |
| `--device` | `auto` selects CUDA when available, otherwise CPU; `cpu`/`cuda` explicit |
| `--threads` | 2 CPU threads, bounded 1..16; CLI process only |
| `--offline` | Require cached official weights; no network |

High mode increases CPU time and memory substantially; no fixed execution times
are promised. Outputs may contain artifacts, and SqueezeNet's compact features
trade quality for lower resource use. Controls weight competing losses rather
than guaranteeing an exact perceptual strength or content preservation score.
Same-seed CPU runs are repeatable in the tested environment; bitwise agreement
across PyTorch versions, devices, or CUDA kernels is not guaranteed.

For repeated use, reuse `NeuralStyleTransfer` from `moodshift.engine.neural_transfer`
and pass frozen `TransferSettings` from `moodshift.engine.settings`. It lazily
loads one model per engine instance. Calls are intended to run sequentially.
Images and files are not mutated; target features are detached, and only output
pixels receive gradients. The library leaves global CPU thread counts unchanged.

### Model provenance and cache

The architecture is `torchvision.models.squeezenet1_1` with
`SqueezeNet1_1_Weights.DEFAULT` (`IMAGENET1K_V1`), trained on ImageNet-1K.
See the [official model documentation](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.squeezenet1_1.html)
and [TorchVision implementation](https://github.com/pytorch/vision/blob/v0.25.0/torchvision/models/squeezenet.py).
The single checkpoint is
`https://download.pytorch.org/models/squeezenet1_1-b8a52dc0.pth` (about 4.7 MB).
PyTorch/TorchVision code is BSD licensed; consult the linked upstream model and
dataset terms for your intended use of pretrained weights. No weights are bundled.

Only intermediate feature indices 1, 3, 6, 9 are used; index 9 also supplies content
features. The classifier is discarded. RGB pixels are scaled to 0..1 and use the
official ImageNet mean/std, without the classification center crop. Style compares
channel correlations normalized by C*H*W; content compares spatial feature maps.
Target-energy scaling balances feature magnitudes, and total variation discourages
pixel noise. Adam runs at learning rate 0.02 with finite-loss/gradient checks,
gradient clipping, and pixel clamping each step. This follows the neural feature
optimization principle in the [official PyTorch tutorial](https://docs.pytorch.org/tutorials/advanced/neural_style_tutorial.html),
using SqueezeNet and Adam rather than that tutorial's VGG/LBFGS choices.

Normal `torch.hub` caching is used; set `TORCH_HOME` to relocate it. To keep the
cache in the ignored repository directory on PowerShell:

```powershell
$env:TORCH_HOME = Join-Path (Get-Location) '.cache\torch'
```

TorchVision checks downloaded hashes, and the loader checks the official SHA256
prefix on cache reuse. Missing offline weights, corrupt checkpoints, download
failures, device/allocation failures, and nonfinite optimization raise errors;
there is no fallback image or untrained model. Corrupt checkpoints are preserved
for manual removal. Reduce resolution or choose CPU after a device/memory failure;
the engine does not silently restart on another device.

### Offline unit tests and real-model integration

The ordinary test command never downloads weights. Engine unit tests use a small
differentiable test extractor solely to isolate optimization behavior. The real
model test is explicitly opt-in and requires the checkpoint already in the cache
(run a normal nonzero CLI transfer once to download it):

```powershell
$env:MOODSHIFT_RUN_INTEGRATION = '1'
python -m pytest -q -s -m integration
Remove-Item Env:MOODSHIFT_RUN_INTEGRATION
```

Real-model validation on Windows CPU loaded the official weights, generated finite
RGB output, and confirmed that changing the reference changes the output. The
8-step generated-image smoke test measured mean absolute channel differences of
19.924 from content and 14.445 between two references (uint8 scale). These are
integration checks, not aesthetic quality benchmarks. Local demo assets are in
ignored `outputs/milestone3/`; they are not committed.
The separate 20-step, 96x80 CLI run reduced the normalized pretrained style error
from 0.539252 to 0.288079, with finite output in [0,1]. CUDA was unavailable in the
installed CPU build and has not been tested on this workstation.

## Training & Experiments tab

Launch the app from the project root using:

```powershell
.venv/Scripts/python.exe -m moodshift.ui.app
```

The Studio retains the original Color Grade, Neural Transfer and Hybrid modes.
A separate **Training & Experiments** tab presents locally available checkpoint
history, loss curve, and held-out content, style and generated images. The trained
checkpoint records its architecture, epoch history and seed; it does not embed the
historical image counts or every training setting. The tab labels the current local
dataset inventory separately from historical facts. Missing evidence is reported.

The experimental decoder is a 250,691-parameter, content-guided model trained with
Adam on CPU at 96 × 96 with batch size 1 for five epochs. The official pretrained
SqueezeNet 1.1 encoder remains frozen: its learned ImageNet features provide a
stable perceptual representation while only decoder weights are updated. Training
used the local photographic content and style folders; validation used separately
held-out folders. The run used 11 content and 11 style training images and two of
each for validation, as recorded during the experiment; these historical counts
are not embedded in the checkpoint. No dataset is included or redistributed.
Cross-split exact byte and decoded-pixel checks were performed; near duplicates and
photo licensing require human assessment.

Experimental inference center-crops both uploads to 96 × 96 and runs the trained
CPU decoder once. Its skip path preserves structure, but style influence is subtle:
on two held-out examples swapping styles changed mean RGB values by about 0.0103 on
the 0–1 scale. This does not establish broad quality or generalization. The original
Neural Transfer mode instead optimizes pixels for each request through frozen
SqueezeNet features, and Hybrid combines that existing approach with a grade.

### Restoring the local experiment

Checkpoints, photographs, and generated evidence are ignored by Git and are not
included with a clone. On another machine, privately copy the **trusted, original**
`experiment_7c2` folder into `checkpoints/experiment_7c2/`, including
`checkpoint.pt`, `losses.csv`, `loss_curve.png`, and `val_*` PNGs if available.
The app accepts only the exact guided checkpoint with SHA-256
`59594434fb728daf2f9dc5cc27ec0e911ce4755819fcb05f54c0cf83b4c5a485`;
other or modified checkpoints are rejected even if their architecture matches.
Only use a copy from a source you trust. Install the project dependencies and
restore the official TorchVision SqueezeNet 1.1 weights to its normal PyTorch cache
(or configure `TORCH_HOME` to the cache containing them). The app checks the
encoder's official checkpoint hash and does not download weights or use random
weights for experimental inference. Restart the app after restoring files. If a
checkpoint, encoder, chart, or example is missing, the tab reports what is available;
experimental generation remains disabled when required weights cannot load.
