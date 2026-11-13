"""Procedural synthetic scene and style references; no photography or AI claims."""

import numpy as np
from PIL import Image

from moodshift.imaging.grading import apply_grade
from moodshift.imaging.grading_parameters import GradeParameters


def synthetic_scene() -> Image.Image:
    """Render a deterministic 320x240 landscape/ramp fixture for color assessment."""
    y, x = np.mgrid[0:240, 0:320].astype(np.float32)
    x /= 319
    y /= 239
    sky = np.stack((0.28 + y * 0.62, 0.50 + y * 0.40, 0.78 + y * 0.18), axis=-1)
    sun = np.exp(-((x - 0.74) ** 2 + (y - 0.26) ** 2) / 0.003)
    rgb = sky * (1 - sun[..., None]) + np.array([1, 0.96, 0.80]) * sun[..., None]
    ridge = 0.49 - 0.13 * np.sin(x * 8) - 0.07 * np.sin(x * 21)
    mountains = y > ridge
    mountain_color = np.stack(
        (0.19 + x * 0.14, 0.29 + x * 0.12, 0.32 + x * 0.11), axis=-1
    )
    rgb = np.where(mountains[..., None], mountain_color, rgb)
    foreground = y > (0.72 + 0.04 * np.sin(x * 9))
    texture = np.sin(x * 180 + y * 40) * np.sin(y * 200) * 0.02
    ground = np.stack(
        (
            0.27 + y * 0.18 + texture,
            0.38 + y * 0.12 + texture,
            0.12 + y * 0.10 + texture,
        ),
        axis=-1,
    )
    rgb = np.where(foreground[..., None], ground, rgb)
    ramp = np.repeat(x[..., None], 3, axis=-1)
    rgb = np.where((y > 0.93)[..., None], ramp, rgb)
    return Image.fromarray(np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8))


def synthetic_style_reference(parameters: GradeParameters) -> Image.Image:
    """Grade a shared 128px procedural color/texture study as a synthetic reference.

    Generated locally from original mathematical patterns; no external assets,
    models, or copyrighted source imagery are used. This is input to the neural
    model, never a substitute for a generated neural result.
    """
    y, x = np.mgrid[0:128, 0:128].astype(np.float32) / 127
    light = 0.12 + 0.74 * x
    wave = 0.045 * np.sin(x * 32) * np.cos(y * 25)
    rgb = np.stack(
        (
            light + wave + 0.08 * np.sin(y * 5),
            light + wave,
            light + wave + 0.08 * np.cos(y * 7),
        ),
        axis=-1,
    )
    image = Image.fromarray(np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    return apply_grade(image, parameters, seed=0)
