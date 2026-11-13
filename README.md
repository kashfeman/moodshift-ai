# MoodShift AI

A university Digital Image Processing project for exploring mood-based image
styling. **Milestone 3 implements real reference-guided neural style transfer.**

Implemented:

- Installable, typed Python package using a `src` layout.
- Immutable configuration with validated limits and default settings.
- Immutable metadata for Cinematic, Sunset, Cold Blue, Vintage, and Dreamy.
- Bounded PNG/JPEG/WebP loading, EXIF orientation, and RGB normalization.
- Aspect-preserving resize, RGB/LAB conversion, and image statistics.
- PNG/JPEG export without overwriting existing files.
- Optimization-based neural transfer with frozen pretrained SqueezeNet features,
  content loss, normalized Gram style loss, total variation, and Adam.
- CPU/CUDA selection, bounded quality controls, and a local command-line demo.
- Generated-image tests, Ruff checks, and GitHub Actions for Python 3.12.

NOT YET IMPLEMENTED:

- Mood preset processing and automatic mood/reference selection.
- Application services coordinating processing and a Gradio user interface.

There is no UI or preset-specific transformation. Presets remain metadata.
Neural output is actually optimized against the supplied reference. No paid API
is used. Only the official SqueezeNet checkpoint downloads on first nonzero use.

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
Pytest and Ruff remain developer dependencies. Gradio is deferred. CUDA use
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

`moodshift.presets.PRESETS` contains the five immutable metadata entries.
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

## Neural transfer and CLI

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
