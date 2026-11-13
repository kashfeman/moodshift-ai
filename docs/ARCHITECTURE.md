# Application architecture

## Implemented through Milestone 4

The installable package uses src-layout and Python >=3.12,<3.13, matching the
Windows 3.12.10 environment and Python 3.12 CI. Preserve the existing environment;
do not reinstall Python or switch global interpreters.

`config.py` owns immutable validated limits/defaults. `presets.py` owns frozen
metadata, typed GradeParameters recipes, and explicit ID lookup. Its only imaging
dependency is a standard-library-only parameter dataclass. There are no duplicate
preset registries or dependencies on PyTorch in preset metadata.

| Imaging module | Responsibility |
| --- | --- |
| `exceptions` | Typed invalid, unsupported, limit, and export errors |
| `validation` | Inclusive dimensions/pixel limits and RGB array contracts |
| `io` | Bounded local decode, EXIF orientation, exclusive PNG/JPEG export |
| `transforms` | RGB normalization over white and LANCZOS downsampling |
| `color` | Independent PIL/NumPy copies and OpenCV float CIE LAB conversion |
| `statistics` | Frozen brightness, contrast, RGB/LAB means and deviations |
| `grading_parameters` | Frozen bounded a/b, tone, chroma, bloom, grain controls |
| `tonal` | Smooth lightness masks and monotonic contrast/gamma curves |
| `effects` | Highlight-only glow and deterministic monochromatic grain |
| `grading` | L-preserving chroma adjustment, complete grade, single RGB interpolation |
| `reference_images` | Original deterministic synthetic scene/reference generators |

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

## Neural engine

The engine uses only the official ImageNet SqueezeNet 1.1 checkpoint via
TorchVision. Runtime dependencies are PyTorch 2.10.x and TorchVision 0.25.x; CPU
builds are preferred on the current Windows workstation and installed in CI.

| Engine module | Responsibility |
| --- | --- |
| `settings` | Frozen, bounded strengths, step count, resolution, seed, quality modes |
| `model_loader` | Actual CPU/CUDA selection, lazy pretrained loading, cache hash checks |
| `features` | Frozen evaluation-mode CNN prefix; intermediate maps at 1/3/6/9 |
| `tensors` | RGB tensor conversion and official ImageNet mean/std normalization |
| `losses` | Content MSE, normalized Gram style MSE, squared total variation |
| `neural_transfer` | Load validated images, resize, prepare targets, optimize pixels |
| `exceptions` | Model, device, nonfinite, and optimization failure contracts |
| `demo` | Argument parsing, process-local CPU threads, startup/completion, safe export |

Current implemented flow: local CLI -> neural transfer -> imaging validation and
downsampling -> pretrained feature extraction -> Adam pixel optimization -> RGB
reconstruction -> existing non-overwriting export. The CLI acts as a composition
entry point. The Milestone 4 service composes this unchanged engine with grading.
Imaging does not depend on engine or services.

The feature boundary permits a tiny differentiable extractor in isolated tests;
production always loads official pretrained weights by default. No random-weight
fallback, alternate large model, or global model singleton exists. Reuse one engine
instance for sequential requests to reuse its extractor. Imports/zero strength
never trigger a model download. Zero strength returns full-size decoded content
pixels, while both input paths still undergo imaging validation.

Nonzero transfer resizes each input independently without cropping or upscaling.
Each axis must remain >=17px. Optimization runs at content processing dimensions,
not original resolution. Input tensors contain RGB in [0,1]; ImageNet normalization
is applied inside feature extraction. Content targets use index 9. Style targets
are per-image Gram matrices FF^T/(C*H*W) at indices 1/3/6/9, so reference dimensions
may differ. Targets are computed without gradients; classifier layers are unused.

Content MSE is divided by target feature mean-square (floor 1e-4). Each style MSE
is divided by its target Gram mean-square (same floor), then averaged over layers.
The total objective is 10*content_preservation*content_loss +
5*style_strength*style_loss + 0.01*TV. These are empirical SqueezeNet defaults, not
perceptual guarantees. Adam optimizes only output pixels with lr=0.02, max gradient
norm 10, and clamping to [0,1] after each step. Tiny seeded initialization noise
uses a local generator. No intermediates are written to disk. Nonfinite values
abort before an output is returned. CUDA availability controls automatic selection;
explicit unavailable CUDA fails. Device/memory failures are reported for caller
retry with smaller images/CPU; no silent partial output or device restart occurs.

Preview is 60 steps/192px; high is 180 steps/384px. Bounds are 1..500 steps and
32..512px maximum side. High mode consumes substantially more memory and CPU time.
CPU results are repeatable for the tested setup; CUDA/cross-version bitwise
determinism is not promised. SqueezeNet texture transfer can introduce artifacts
and does not constitute a specialized mood-transfer model.

Official weight provenance, upstream sources, cache location, and failure recovery
are documented in README. Normal PyTorch caching honors TORCH_HOME; the local
verification cache is ignored `.cache/torch`. Corrupt weights are never deleted
automatically. TorchVision verifies downloads; cached reuse checks the SHA256
prefix from the official checkpoint filename.

## Mood service and hybrid grading

| Service module | Responsibility |
| --- | --- |
| `models` | Frozen validated requests, effective settings, result metadata/image |
| `references` | Validate selection and manage temporary synthetic reference lifetime |
| `mood_transfer` | Orchestrate grading, neural transfer, optional post-grade, final blend |
| `demo` | Safe mode-aware CLI with honest backend/device reporting |
| `examples` | Reproducible synthetic fixture, comparison sheet, optional real hybrid |

Dependency flow: service -> engine/imaging; engine -> imaging. Grade-only service
imports the engine's standard-library-only settings dataclass, never PyTorch or
TorchVision. The actual NeuralStyleTransfer import/creation is lazy and reused
across sequential calls. Engine APIs and previous defaults remain unchanged.

Grade mode requires a preset and rejects references. Neural mode accepts either
a user reference or a preset's synthetic reference, without post-grading. Hybrid
requires a preset, runs real neural transfer, then applies that preset's grade;
a supplied reference overrides the synthetic one. Ambiguous unused controls are
rejected rather than silently ignored.

All input files use existing validation and EXIF/RGB normalization. Built-in
references are deterministic 128px mathematical palette/texture patterns graded
from the single preset recipe registry. Only the small reference PNG is written
to a temporary directory for the path-based neural API; it is cleaned up even on
failure. No external assets or model-generated reference claims are involved.
See assets/presets/README.md for provenance.

Photographic grades partition normalized L with smoothstep masks. Shadow/midtone/
highlight a/b offsets taper near tone endpoints; saturation/vibrance act on chroma,
leaving L untouched until explicit tone curves. Vibrance reduces boosts for warm
hues heuristically. Sigmoid contrast and gamma are monotonic, followed by black
lift/white compression. Dreamy uses a Gaussian bright-pixel halo with screen-like
blending; Vintage adds delicate shared-channel midtone grain with a local RNG.
LAB conversion may still clip colors outside the RGB gamut. Color management/HDR
limitations remain those of the imaging foundation.

Intensity 0..100 is one final encoded-RGB blend between content and full processed
appearance. Nonzero neural stages use engine style_strength=1. Zero bypasses all
effects, model construction, reference generation, and resize, returning exact
normalized content pixels. Supplied input paths and mode combinations remain
validated. No inference failure is replaced with a grade-only or unchanged image.

Service qualities are preview=20 steps/128px, balanced=60/256px, high=120/384px.
Explicit steps/resolution obey existing engine bounds. Quality resolution limits
neural computation; default final dimensions remain the original's by resampling
neural pixels back to the original grid. This retains layout, not fine detail.
Explicit max_size also limits final output. No cropping, stretching of source
content, or spatial warp is used. Grade-only stays full-size unless explicitly
limited; its metadata always reports zero neural steps.

Frozen result metadata records the requested mode/preset, actual engine/device,
neural/grading execution flags, consumed reference source, original/internal/final
sizes, and effective controls. Zero reports engine/reference none. Grade-only is
numpy-opencv on CPU. Injected unit-test backends are explicitly marked rather than
claimed to be pretrained. Neural CPU/CUDA failures propagate; no fake progress or
fixed-time estimates exist. The CLI checks output availability early and still
uses exclusive export against race-time overwrites.

## NOT YET IMPLEMENTED

The UI, web server, and automatic mood/reference recognition remain unimplemented.

| Package | Planned responsibility | Future dependencies |
| --- | --- | --- |
| `ui` | Collect inputs, call services, present results and errors | Gradio, services |

Future flow: UI -> services -> validated imaging -> neural engine -> image export.
UI must not contain processing or neural logic. Imaging and engine must not
import UI/services. Add interfaces only for concrete integration needs. Preset
processing is implemented through the service and grading modules. No paid APIs,
UI, or web server are included. No weights are bundled with the repository.

## Assets and verification

Tracked preset assets belong in `assets/presets/`; generated images go in ignored
`outputs/`; future weights go in ignored `weights/`.

Generated-image tests cover formats, corrupt/unsupported input, limits, bomb
protection, EXIF, alpha, array ownership/channel order, aspect ratio, LAB tolerance,
known statistics, export/reload, and original-file preservation. CI installs the
package on Python 3.12, runs Ruff lint/format checks and Pytest. Engine unit tests
use deterministic tensors and a differentiable test extractor, never downloads.
An opt-in integration marker tests actual cached pretrained features and reference
influence; it skips if not explicitly enabled or weights are absent. Real CPU
inference has also been exercised through the CLI. README documents development
commands and safe cache diagnostics without administrator access.

Milestone 4 tests add all-preset effects/distinctness, smooth masks, exact zero,
chroma/channel correctness, seeded grain, immutable inputs, strict controls,
temporary reference cleanup, no-PyTorch grading in a subprocess, and mode-specific
neural invocation. Only the neural boundary is mocked in service unit tests. Two
opt-in service integrations exercise actual cached weights for neural/hybrid modes,
separate from the original engine smoke test. The examples command produces real
local grade outputs and an optional cached-model hybrid; it never fabricates a
successful neural result. Synthetic fixtures are not a substitute for future
evaluation on diverse real photographs.
