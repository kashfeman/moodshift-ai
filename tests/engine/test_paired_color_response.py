"""Tests for optional same-content paired-reference training."""

from pathlib import Path

import pytest
import torch
from PIL import Image
from torchvision.models import squeezenet1_1

from moodshift.engine.color_conditioned_decoder import (
    ColorConditionedDecoder,
    color_objective,
    color_step,
)
from moodshift.engine.dataset import (
    PairedReferenceDataset,
    make_paired_reference_loader,
)
from moodshift.engine.features import SqueezeNetFeatures
from moodshift.engine.training import EncoderAdapter


def test_paired_objective_gradients_and_disabled_behavior():
    """Check the paired gradient, frozen encoder, update and disabled parity."""
    torch.manual_seed(23)
    encoder = EncoderAdapter(SqueezeNetFeatures(squeezenet1_1(weights=None).features))
    model = ColorConditionedDecoder()
    content = torch.rand(1, 3, 32, 32)
    style_a = torch.rand(1, 3, 32, 32) * 0.4
    style_b = torch.rand(1, 3, 32, 32) * 0.4 + 0.6
    before = [p.detach().clone() for p in encoder.parameters()]
    default = color_objective(encoder, model, content, style_a, rgb_weight=1)
    disabled = color_objective(
        encoder,
        model,
        content,
        style_a,
        rgb_weight=1,
        paired_style=style_b,
        paired_weight=0,
    )
    assert default.keys() == disabled.keys()
    for name in default:
        torch.testing.assert_close(default[name], disabled[name], rtol=0, atol=0)
    same = color_objective(
        encoder, model, content, style_a, paired_style=style_a, paired_weight=1
    )
    different = color_objective(
        encoder, model, content, style_a, paired_style=style_b, paired_weight=1
    )
    assert same["paired_color_response_loss"].item() == 0
    assert different["paired_color_response_loss"].item() > 0
    different["paired_color_response_loss"].backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()
    )
    assert all(p.grad is None for p in encoder.parameters())
    old = [p.detach().clone() for p in model.parameters()]
    result = color_step(
        encoder,
        model,
        torch.optim.Adam(model.parameters(), lr=1e-4),
        content,
        style_a,
        paired_style=style_b,
        paired_weight=1,
    )
    assert all(torch.isfinite(torch.tensor(value)) for value in result.values())
    assert any(
        not torch.equal(a, b) for a, b in zip(old, model.parameters(), strict=True)
    )
    assert all(
        torch.equal(a, b) and b.grad is None
        for a, b in zip(before, encoder.parameters(), strict=True)
    )
    for kwargs in (
        {"paired_weight": -1},
        {"paired_weight": float("nan")},
        {"response_target": 1.1},
        {"paired_weight": 1},
    ):
        with pytest.raises(ValueError):
            color_objective(encoder, model, content, style_a, **kwargs)


def test_pairing_reproducible_and_split_safe(tmp_path: Path):
    """Keep both references reproducible and within their own split."""
    for split, offset in [("train", 0), ("val", 100)]:
        for role in ("content", "style"):
            folder = tmp_path / split / role
            folder.mkdir(parents=True)
            for i in range(3):
                Image.new(
                    "RGB",
                    (32, 32),
                    (offset + i * 20, 30 if role == "content" else 90, 120),
                ).save(folder / f"{i}.png")
    a = PairedReferenceDataset(tmp_path, "train", seed=5)
    b = PairedReferenceDataset(tmp_path, "train", seed=5)
    v = PairedReferenceDataset(tmp_path, "val", seed=5)
    assert a.base.style == b.base.style and a.second_style == b.second_style
    assert all(x != y for x, y in zip(a.base.style, a.second_style, strict=True))
    assert all("train" in p.parts for p in (*a.base.style, *a.second_style))
    assert all("val" in p.parts for p in (*v.base.style, *v.second_style))
    first = list(make_paired_reference_loader(tmp_path, "train", seed=5))
    second = list(make_paired_reference_loader(tmp_path, "train", seed=5))
    for x, y in zip(first, second, strict=True):
        for left, right in zip(x, y, strict=True):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
