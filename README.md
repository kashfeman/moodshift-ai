# MoodShift AI

A university Digital Image Processing project for exploring mood-based image
styling. **Milestone 1 implements the project foundation only.**

Implemented:

- Installable, typed Python package using a `src` layout.
- Immutable configuration with validated limits and default settings.
- Immutable metadata for Cinematic, Sunset, Cold Blue, Vintage, and Dreamy.
- Unit tests, Ruff configuration, and GitHub Actions for Python 3.11.

Planned for later milestones:

- Image decoding, validation, and classical processing with Pillow, NumPy, OpenCV.
- Neural network components and inference with PyTorch and TorchVision.
- Application services coordinating processing and a Gradio user interface.

There is no runnable UI, image transformation, or AI inference in this milestone.
Preset descriptions express intended styles, not implemented algorithms. No model
weights are downloaded, and no paid APIs are used or required.

## Development

Use Python 3.11. From the repository root on PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check .
python -m ruff format --check .
```

On POSIX shells, activate with `source .venv/bin/activate` instead. Runtime
dependencies are intentionally empty; only Pytest and Ruff are included in the
development extra. Heavy image, ML, and UI dependencies will be introduced when
their functionality is implemented.

## Configuration and presets

`moodshift.config.AppConfig` defaults to a 10 MiB encoded upload limit, maximum
width and height of 4096 pixels, and 16 million decoded pixels. All limits are
inclusive and must be enforced together by future image validation. The default
preset ID is `cinematic`, strength is `0.5` (allowed range `[0, 1]`), and output
format is `PNG` (`JPEG` is also accepted). Configuration validates setting values;
it does not inspect uploads. Custom preset IDs are accepted by configuration;
services will resolve them against the catalog using `get_preset`.

`moodshift.presets.PRESETS` contains the five immutable metadata entries.
`get_preset(id)` raises `KeyError` for unknown IDs.

See [the architecture plan](docs/ARCHITECTURE.md) for module boundaries. Store
future tracked preset assets in `assets/presets/`; keep generated images in the
ignored `outputs/` directory and downloaded weights in ignored `weights/`.
