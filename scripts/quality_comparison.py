"""Compare actual cached CNN reconstruction at 128/256/384 pixels and 20/60 steps.

Use a new output directory. No weights are downloaded. Timing includes the first
lazy model load; CPU threads are fixed to two for this experiment only.
The detail crop is specific to the investigated 978x1365 landscape.
"""


def main() -> None:
    """Generate real outputs, labeled panels, and numerical diagnostics."""
    import hashlib
    import json
    import os
    import time
    from pathlib import Path

    os.environ.setdefault("TORCH_HOME", str(Path.cwd() / ".cache/torch"))
    import argparse

    import cv2
    import numpy as np
    import torch
    from PIL import Image, ImageDraw

    from moodshift.engine.neural_transfer import NeuralStyleTransfer
    from moodshift.engine.settings import TransferSettings
    from moodshift.imaging.grading import apply_grade, blend_images
    from moodshift.imaging.io import load_image
    from moodshift.imaging.reconstruction import reconstruct_hybrid
    from moodshift.presets import get_preset
    from moodshift.services.references import reference_path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out, source = args.output, args.source
    out.mkdir(parents=True, exist_ok=False)
    content = load_image(source)
    content.save(out / "original.png")
    torch.set_num_threads(2)
    engine = NeuralStyleTransfer(device="cpu", allow_download=False)
    preset = get_preset("dreamy")
    rows = []

    def detail_error(image: Image.Image) -> float:
        """Measure source-relative RGB high-frequency error; not aesthetic quality."""
        a = np.asarray(image, dtype=np.float32)
        b = np.asarray(content, dtype=np.float32)
        return float(
            np.mean(
                np.abs(
                    (a - cv2.GaussianBlur(a, (0, 0), 2))
                    - (b - cv2.GaussianBlur(b, (0, 0), 2))
                )
            )
        )

    for resolution in (128, 256, 384):
        for steps in (20, 60):
            start = time.perf_counter()
            with reference_path(None, preset) as ref:
                raw = engine.transfer(
                    source,
                    ref,
                    TransferSettings(
                        steps=steps,
                        max_size=resolution,
                        style_strength=0.5,
                        content_preservation=0.7,
                        seed=0,
                    ),
                )
            elapsed = time.perf_counter() - start
            raw.save(out / f"raw-{resolution}-{steps}.png")
            old = blend_images(
                content,
                apply_grade(
                    raw.resize(content.size, Image.Resampling.LANCZOS), preset.grading
                ),
                0.7,
            )
            new = blend_images(
                content,
                apply_grade(reconstruct_hybrid(content, raw), preset.grading),
                0.7,
            )
            old.save(out / f"old-{resolution}-{steps}.png")
            new.save(out / f"new-{resolution}-{steps}.png")
            row = dict(
                resolution=resolution,
                steps=steps,
                neural_size=raw.size,
                seconds=round(elapsed, 3),
                old_detail_error=detail_error(old),
                new_detail_error=detail_error(new),
            )
            rows.append(row)
            print(row, flush=True)
    for preset_id in ("cinematic", "sunset"):
        blend_images(
            content, apply_grade(content, get_preset(preset_id).grading), 0.7
        ).save(out / f"{preset_id}-grade.png")
    raw = Image.open(out / "raw-256-60.png")
    blend_images(content, raw.resize(content.size, Image.Resampling.LANCZOS), 0.7).save(
        out / "neural-transfer.png"
    )
    items = [
        ("Original", content),
        ("Previous 128 / 20", Image.open(out / "old-128-20.png")),
        ("Guided 128 / 20", Image.open(out / "new-128-20.png")),
        ("Guided 128 / 60", Image.open(out / "new-128-60.png")),
    ]
    for crop in (False, True):
        panel = Image.new("RGB", (4 * 330, 500), (20, 25, 35))
        draw = ImageDraw.Draw(panel)
        for i, (label, img) in enumerate(items):
            img = img.crop((560, 790, 890, 1250)) if crop else img.copy()
            img.thumbnail((326, 460))
            panel.paste(img, (i * 330, 30))
            draw.text((i * 330 + 8, 8), label, fill="white")
        panel.save(
            out / ("detail-comparison.png" if crop else "quality-comparison.png")
        )
    (out / "quality-results.json").write_text(
        json.dumps(
            dict(
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                source_size=content.size,
                style_strength=0.5,
                intensity=70,
                content_preservation=0.7,
                seed=0,
                device="cpu",
                threads=2,
                experiments=rows,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
