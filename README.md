# MoodShift AI

A university Digital Image Processing project for exploring mood-based image
styling. **Milestone 2 implements the image processing foundation.**

Implemented:

- Installable, typed Python package using a `src` layout.
- Immutable configuration with validated limits and default settings.
- Immutable metadata for Cinematic, Sunset, Cold Blue, Vintage, and Dreamy.
- Bounded PNG/JPEG/WebP loading, EXIF orientation, and RGB normalization.
- Aspect-preserving resize, RGB/LAB conversion, and image statistics.
- PNG/JPEG export without overwriting existing files.
- Generated-image tests, Ruff checks, and GitHub Actions for Python 3.12.

NOT YET IMPLEMENTED:

- Reference-guided deep learning mood transfer and mood transformations.
- Neural network components and inference with PyTorch and TorchVision.
- Application services coordinating processing and a Gradio user interface.

There is no runnable UI or AI inference in this milestone.
Preset descriptions express intended styles, not implemented algorithms. No model
weights are downloaded, and no paid APIs are used or required.

## Development

Use Python >=3.12,<3.13 (workstation: 3.12.10). Preserve the existing virtual
environment; do not reinstall Python or switch global interpreters. From the root:

```powershell
.venv\Scripts\Activate.ps1
python --version
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
>=4.11,<5. Pytest and Ruff remain developer dependencies; ML and UI are deferred.

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
