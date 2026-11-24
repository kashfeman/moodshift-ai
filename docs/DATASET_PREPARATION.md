# Preparing training photographs

This milestone prepares data loading only. No photographs are included, downloaded,
or used to train the decoder. Obtain real photographs you have permission to use.

From the project root, place files directly in these four folders:

```text
data/
  train/content/   # photographs whose structure the model should preserve
  train/style/     # photographs providing appearances to learn
  val/content/     # separate content photographs for validation
  val/style/       # separate style photographs for validation
```

Create the folders if needed. Do not put the same photograph in training and
validation, including re-encoded versions. Keep related near-duplicates in the same
split. The `data/` directory is ignored by Git; keep your own backups.

For a first smoke test, aim for at least 10 different training content photos,
10 training style photos, and 2 different validation photos in each category.
These small starter targets do not establish model quality or statistical validity;
more diverse and independently sourced photographs will be needed for evaluation.
JPG, JPEG, PNG and WebP are accepted (extension must match the actual format).
The existing image limits apply: at most 10 MiB, 4096 pixels per side and 16 million
pixels; animated, corrupt and unsupported images are rejected. RGB conversion
composites transparency over white. Images are center-cropped and resized with
aspect ratio preserved; small images may be enlarged.

From PowerShell in the project root with the project environment installed, run:

```powershell
.venv/Scripts/python.exe -m moodshift.engine.dataset --data data
.venv/Scripts/python.exe -m moodshift.engine.dataset --data data --smoke --size 96 --batch-size 4 --seed 0
```

Use `--size 128` for 128-pixel batches. The command lists valid counts and reports
missing folders, unsupported extensions, invalid images, and byte-identical files
across splits. It returns a nonzero exit code if starter targets or integrity checks
are not met. Hashing does not detect visually similar images or re-encodings; check
those manually. The smoke test loads one CPU batch per split and does not train.
A dataset can be loaded with fewer images than the starter targets, but is not
reported ready. Invalid files found during scanning are omitted from the dataset;
inspect the validation report before use. If a file changes or becomes corrupt
between scanning and loading, loading fails with its path instead of silently
substituting another image.

For Python use, call `make_data_loader("data", "train", size=96, batch_size=4,
seed=0)` or use split `"val"`. Each batch contains `(content, style)` tensors of
shape `N x 3 x size x size` in `[0, 1]`. Every content image appears once per pass;
seeded styles cycle if the style set is smaller. Pair assignments are fixed for a
given seed, training order is seeded and shuffled, and validation order is fixed.
Use separate loader instances with the same seed to reproduce the same sequence.

## CPU decoder training

The experimental training CLI uses the cached official ImageNet SqueezeNet 1.1
weights; it fails if they are unavailable or fail the existing checksum check.
It never substitutes random encoder weights. Validate your photographs first:

```powershell
.venv/Scripts/python.exe -m moodshift.engine.train_decoder --data data --output checkpoints/my_run --epochs 5 --size 96 --batch-size 1 --learning-rate 0.0001 --seed 0 --threads 2 --device cpu
```

The output directory must not already exist. Each completed epoch saves
`checkpoint.pt` with decoder, optimizer, settings and history, plus `losses.csv`
and `loss_curve.png`. At the end the CLI reloads the saved checkpoint and writes
before, style, and after images from the validation set. To resume, provide
`--resume checkpoints/my_run/checkpoint.pt` and a new output directory, increase
`--epochs` to the desired total, and keep size, batch size, seed, and learning rate
the same. Keep the dataset unchanged for a meaningful continuation. Checkpoints
must come from a trusted source. An interrupted epoch is not checkpointed.

The training command refuses failed starter checks, byte-identical cross-split
files, and cross-split images with identical decoded RGB pixels. It cannot detect
near-duplicates or the same photograph saved with lossy changes; inspect the
splits manually. Validation does not update model parameters. A small dataset and
short CPU run are an initial experiment, not a demonstration of generalization.

The current objective normalizes the feature losses and adds an RGB identity
reconstruction loss; decoder gradients are clipped to a norm of 1. The objective
changed after the first experiment, so its old checkpoints cannot be resumed with
the new objective and their loss values are not directly comparable. The new
checkpoint format records the objective version. Both checkpoints can still be
loaded for inference with `ImageDecoder`.

## Experimental content-guided decoder

`python -m moodshift.engine.train_guided --data data --output checkpoints/guided_run --epochs 5 --seed 0 --previous checkpoints/experiment_7c1`
starts a new CPU-only experiment with cached official frozen encoder weights,
96-pixel images, batch size 1, Adam at 0.0001, and clipped gradients. The output
directory must not exist. The optional previous run must contain matching saved
validation inputs; the command checks their RGB pixels before generating comparison
panels. Leave out `--previous` to train without a panel. The new checkpoint uses
`GuidedDecoder`, not `ImageDecoder`; the two state dictionaries are incompatible.

The guided decoder uses content RGB as a full-resolution logit skip and predicts
bounded, learned RGB shifts from AdaIN-aligned SqueezeNet features and content.
Its loss combines normalized feature and style statistics, identity RGB MSE
(weight 10), and stylized-image edge L1 (weight 2). The skip preserves edges by
design, so visual similarity to content alone does not prove effective style
transfer. Measure output changes when swapping style references. Results can look
close to the input and require broader evaluation; this experiment is not connected
to the application. The command saves epoch metrics and a reloadable checkpoint but
does not support resuming training.

### Controlled loss-weight comparison

Both commands below start from newly initialized decoder weights. With the same
seed, unchanged dataset and official encoder checkpoint, software versions, and
CPU settings, they use the same initialization and pairing order. Use separate,
not-yet-existing output directories; neither run resumes an older checkpoint.

```powershell
.venv/Scripts/python.exe -m moodshift.engine.train_guided --data data --output checkpoints/step8b_baseline --epochs 5 --seed 0 --loss-preset baseline
.venv/Scripts/python.exe -m moodshift.engine.train_guided --data data --output checkpoints/step8b_stronger_style --epochs 5 --seed 0 --loss-preset stronger-style
```

Baseline content/style/identity/edge weights are 1/1/10/2; stronger-style uses
1/3/5/1. Individual nonnegative finite values may be specified using
`--content-weight`, `--style-weight`, `--identity-weight`, and `--edge-weight`;
at least one must be positive. Overrides modify the selected preset. Each run
saves its actual weights and settings in `experiment.json` and `checkpoint.pt`.
Compare `initial_decoder_sha256`, `dataset_sha256`, `encoder_checkpoint_sha256`,
seed, epochs, optimizer and environment before interpreting results. These
fingerprints help detect differences; they are not a guarantee of bitwise
reproducibility across systems or changes during a run. The CSV retains unweighted
component losses; total losses across presets use different weights and are not
directly comparable. No runs have been started by this configuration change.

### Reference-conditioned experiment (not yet trained)

A separate `style-film-v1` decoder receives the actual style image's SqueezeNet
layer-9 channel mean and standard deviation. A small MLP applies bounded feature
modulation and a bounded RGB color contribution to the learned residual. Its
full-resolution content shortcut and baseline loss terms remain in place. It is
not a drop-in replacement for `guided-v1` checkpoints and is not connected to UI.
Only a one-step training smoke test has been performed; reference conditioning
is connected, but improved transfer, sharpness, and color quality are not yet
established. The encoder remains frozen.

For a controlled future five-epoch run against the existing Step 8B baseline,
use the same unchanged dataset, cached encoder, seed and baseline loss weights:

```powershell
.venv/Scripts/python.exe -m moodshift.engine.train_guided --data data --output checkpoints/step8d_style_film --epochs 5 --seed 0 --loss-preset baseline --architecture style-film-v1
```

This starts from scratch and creates a new directory. Under the same seed, the
shared convolution layers use the same initialization as the baseline; additional
conditioning parameters are new. After a real completed run, compare identical
validation content/style pairs, including swapped references, with:

```powershell
.venv/Scripts/python.exe -m moodshift.engine.compare_conditioned --baseline checkpoints/step8b_baseline/checkpoint.pt --conditioned checkpoints/step8d_style_film/checkpoint.pt --data data --output outputs/step8d_comparison
```

The comparison refuses mismatched dataset, encoder, seed, training settings, loss
weights or shared-layer initialization, and never reuses an output directory.
It writes RGB and edge preservation, layer-9 style statistics, style-swap change,
and comparison panels. Metrics on two validation photographs cannot establish
perceptual quality or generalization; inspect outputs for color shifts, artifacts,
and content detail. Existing guided checkpoints and the Studio remain unchanged.

### RGB-reference-conditioned experiment

`color-film-v1` extends `style-film-v1` with a small trainable branch fed the
reference image's RGB channel means and standard deviations. It retains the
feature-conditioning path and content shortcut, and uses the same baseline
training objective. The new branch does not itself impose a direct RGB matching
loss, so it can be ignored by training. This architecture is separate from existing
checkpoints and is not integrated into Studio.

Step 8G trained five CPU epochs with the baseline preset, seed 0, 96px and batch
size one. Its initialized shared style-film parameters, dataset fingerprint and
encoder hash match Step 8E. On two held-out content images and two style references,
the measured style-swap response was weaker than Step 8E, so Step 8E remains the
preferred experimental checkpoint. Inspect local `outputs/step8g_comparison/`
metrics and images; these local photographic examples are not redistribution-cleared.

### Reference RGB color-loss experiment

`color-film-v1` also supports the opt-in
`--reference-color-loss-weight` option. A weight of 1 adds the MSE between
per-channel RGB means plus the MSE between stabilized population standard
deviations of output and reference, with RGB in [0,1]. It keeps the original
content, deep-style, identity and edge losses. Weight zero is the Step 8G default.
The coefficient is recorded in experiment metadata and the loss is reported
separately in the CSV. Training starts from scratch in a new output directory.

Step 8H used weight 1, baseline loss weights, seed 0 and five CPU epochs. Its
full initial decoder, data and encoder fingerprints match Step 8G. On the two
original validation pairs, Step 8H did not lower its average RGB color-statistics
error compared with Step 8G or Step 8E. The sample is too small to establish
quality or generalization. The earlier Step 8E remains the preferred checkpoint
for reference sensitivity; there is no Studio integration for these models.

## Paired-reference color response experiment

`train_guided` can optionally train `color-film-v1` using a second, byte-distinct
style image from the same split. The original pair ordering is retained. The
second reference is selected reproducibly using the epoch seed. Validation pairs
use only validation images; byte-identical cross-split files are rejected by the
existing split validation. At least two byte-distinct style files are required in
each split when this option is enabled.

The mean RGB response loss is `MSE(mean(output_a) - mean(output_b),
response_target * (mean(style_a) - mean(style_b)))`. Its weight defaults to zero,
which preserves the existing training path. The response target defaults to 0.25
and must be between 0 and 1. Both settings are recorded in experiment metadata
and checkpoints. The paired loss is logged separately. Weighted totals from runs
using different objectives are not directly comparable. This experimental loss
may produce unnatural recoloring and needs held-out visual evaluation.
