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
