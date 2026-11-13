# Synthetic style reference provenance

Built-in presets use locally generated **synthetic style references**, not external
photographs or AI-generated outputs. The source is
`src/moodshift/imaging/reference_images.py:synthetic_style_reference`.

The original recipe renders a 128x128 neutral-to-colored lightness ramp with
deterministic sine/cosine color fields and texture. It applies the selected
immutable GradeParameters from `moodshift.presets.PRESETS`, using fixed seed 0.
Thus references share structure but have distinct preset palettes and tone curves.
No asset downloads, borrowed photographs, external API, or extra model is used.

For neural/hybrid requests without a user reference, the service encodes one
temporary PNG, passes it to the real pretrained neural engine, then removes the
temporary directory. The reference is a model input, never represented as a
model result. User references override it in hybrid mode. Grade mode does not
create or apply a style reference at all.

The independent `synthetic_scene` function renders an original 320x240 landscape
with a grayscale ramp for reproducible tests and visual comparisons. It is also
explicitly synthetic, not a photograph. The examples CLI writes fixtures and
outputs under a new caller-selected directory; use ignored `outputs/` directories
for local artifacts. Pretrained SqueezeNet provenance remains in the project
README and is separate from these reference assets.
