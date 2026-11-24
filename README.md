# MoodShift AI

MoodShift AI is a local Python application for photographic mood editing. Studio
provides Color Grade, Neural Transfer and Hybrid modes with five presets. Neural
Transfer optimizes an image using frozen pretrained SqueezeNet 1.1 features; Hybrid
combines neural color transfer with detail-preserving reconstruction and grading.

**Reference Mood Boost** is an optional conventional CIELAB photographic color
transfer stage in Neural and Hybrid. It uses an uploaded reference after neural
processing. It is not neural training. The separate Training & Experiments tab
shows recorded training history and offers a distinct 96×96 trained-decoder
preview when its trusted local checkpoint is installed.

## Setup (Windows PowerShell)

Install Python 3.12, open PowerShell in the project folder, and run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m moodshift.ui.app
```

Open <http://127.0.0.1:7860>. The app runs locally; stop it with Ctrl+C.
Installation needs access to the declared packages in `pyproject.toml`. Neural
features also require the official pretrained SqueezeNet weights. The UI defaults
to offline mode; disable it to allow their official first download, or provide a
previously cached official TorchVision checkpoint. No model download is needed
for Color Grade. CPU neural optimization can take time.

## Studio

1. Upload an RGB photograph (PNG, JPEG or WebP) and choose a mode and preset.
2. For Neural or Hybrid, optionally upload a neural style reference. Without one,
   a labeled synthetic preset reference is used by the neural engine.
3. To use Reference Mood Boost, open its section, enable it, upload a separate
   boost reference and choose a strength. Start around 50. Without that upload,
   or at strength 0, the boost does not change the result.
4. Generate, use the before/after slider, and save the lossless PNG. Output keeps
   the source dimensions unless an output limit is selected.

The boost is applied only when neural processing actually runs. It does not alter
the trained checkpoint. High strength can create unrealistic color or skin tones;
its skin protection is heuristic. Neural mode blends neural texture with a
content-guided reconstruction to reduce blur; some texture is softened and
artifacts may remain. Results vary with reference and settings.

## Experimental trained decoder

The dashboard expects the separately packaged trusted checkpoint at
`checkpoints/experiment_7c2/checkpoint.pt`. Extract the trusted checkpoint archive
into the project root, preserving its paths. Only that project's pinned checkpoint
is accepted; the application verifies its SHA-256 and checks its architecture and
history. It also requires cached official SqueezeNet weights. If files are missing,
the dashboard explains what is unavailable; Studio still works.

The guided decoder was trained for five CPU epochs on a small local photograph
dataset. It preserves structure but produces subtle style changes at 96×96; this
is not proof of broad generalization. The experimental tab also transfers its
smoothed color changes onto the matching full-resolution center crop, preserving
original detail. A bounded slider amplifies only the predicted color change;
held-out examples use 1.5× when the matching original photos are available.
This postprocessing is not high-resolution decoder inference.
Both the native result and reconstruction have separate PNG downloads. Training
data and photographs are not
included. The source ZIP includes actual numerical records and a loss graph;
validation photographs and derivatives require permission before distribution.
See `THIRD_PARTY_NOTICES.md` for dependency and model notices. Rights to distribute
training data and trained weights need confirmation; the separate checkpoint ZIP
should only be shared when authorized.

## Checks and training code

```powershell
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
python -m pip check
```

Training and dataset utilities are in `src/moodshift/engine`. See
`docs/DATASET_PREPARATION.md` before preparing your own licensed photographs.
The program does not automatically download training datasets. The application
and tests do not require the local dataset for ordinary use.
