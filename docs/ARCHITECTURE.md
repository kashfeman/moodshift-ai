# Application architecture

## Implemented through Milestone 2

The installable package uses src-layout and Python >=3.12,<3.13, matching the
Windows 3.12.10 environment and Python 3.12 CI. Preserve the existing environment;
do not reinstall Python or switch global interpreters.

`config.py` owns immutable validated limits/defaults. `presets.py` owns frozen
metadata and explicit ID lookup. Neither imports imaging or machine learning.

| Imaging module | Responsibility |
| --- | --- |
| `exceptions` | Typed invalid, unsupported, limit, and export errors |
| `validation` | Inclusive dimensions/pixel limits and RGB array contracts |
| `io` | Bounded local decode, EXIF orientation, exclusive PNG/JPEG export |
| `transforms` | RGB normalization over white and LANCZOS downsampling |
| `color` | Independent PIL/NumPy copies and OpenCV float CIE LAB conversion |
| `statistics` | Frozen brightness, contrast, RGB/LAB means and deviations |

Loading checks file presence and encoded size, reads at most the byte limit plus
one, validates raw header dimensions before decoding, rejects multiple frames,
verifies structure, then decodes the same bounded bytes. Oriented dimensions are
checked before rotation; PNG EXIF access can trigger decoding after raw checks. EXIF
orientation is applied once and returned RGB pixels are detached from the file.
Pillow bomb protections remain enabled; warnings/errors become domain errors.
PNG, JPEG, and WebP are detected from content, not filename extension.

Transparency composites over white; grayscale becomes RGB. High-bit-depth/float
modes are rejected, ICC profiles are not interpreted, and metadata is removed.
Basic Pillow CMYK conversion is supported. This is deterministic 8-bit processing,
not archival color management. Resizing never upscales and allows pixel rounding.
Operations do not mutate caller images or arrays.

Arrays are H x W x 3 uint8 RGB. LAB is float32 CIE units (L 0..100, signed a/b,
D65), not packed uint8 LAB. Reverse conversion rounds and clips out-of-gamut RGB.
Brightness/contrast are mean/population deviation of gamma-encoded BT.709 luma
in 0..1, not physical exposure estimates. Channel statistics use population std.

Export encodes before exclusive creation, preserving existing files and symlink
targets. PNG is lossless; JPEG defaults to quality 95 without chroma subsampling
and is lossy. Parents must exist. Failed writes may leave a partial new file.
Callers select local paths; no server path containment or process isolation is
provided. Encoded size and pixel limits reduce allocation risk, not all decoder
security risks.

## NOT YET IMPLEMENTED

`engine`, `services`, and `ui` contain documentation only.

| Package | Planned responsibility | Future dependencies |
| --- | --- | --- |
| `engine` | Reference-guided neural transfer, weight loading, inference | PyTorch, TorchVision |
| `services` | Resolve presets and coordinate imaging/inference use cases | Config, presets, imaging, engine |
| `ui` | Collect inputs, call services, present results and errors | Gradio, services |

Future flow: UI -> services -> validated imaging -> neural engine -> image export.
UI must not contain processing or neural logic. Imaging and engine must not
import UI/services. Add interfaces only for concrete integration needs. Preset
metadata does not implement mood transformations. No fake inference, pretrained
weights, paid APIs, UI, or web server is included.

## Assets and verification

Tracked preset assets belong in `assets/presets/`; generated images go in ignored
`outputs/`; future weights go in ignored `weights/`.

Generated-image tests cover formats, corrupt/unsupported input, limits, bomb
protection, EXIF, alpha, array ownership/channel order, aspect ratio, LAB tolerance,
known statistics, export/reload, and original-file preservation. CI installs the
package on Python 3.12, runs Ruff lint/format checks and Pytest. README documents
development commands and safe cache diagnostics without administrator access.
