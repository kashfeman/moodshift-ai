"""Offline unit tests for the separate reference-conditioned decoder."""

import torch
from torchvision.models import squeezenet1_1

from moodshift.engine.features import SqueezeNetFeatures
from moodshift.engine.guided_decoder import GuidedDecoder
from moodshift.engine.style_conditioned_decoder import (
    StyleConditionedDecoder,
    conditioned_step,
)
from moodshift.engine.train_guided import model_fingerprint, shared_baseline_fingerprint
from moodshift.engine.training import EncoderAdapter, align_features


def test_reference_conditioning_gradients_and_frozen_encoder() -> None:
    """Hold content/alignment fixed and vary only actual reference features."""
    torch.manual_seed(31)
    encoder = EncoderAdapter(SqueezeNetFeatures(squeezenet1_1(weights=None).features))
    decoder = StyleConditionedDecoder()
    content = torch.rand(1, 3, 32, 40)
    style = torch.rand(1, 3, 32, 40)
    other = torch.rand(1, 3, 32, 40)
    with torch.no_grad():
        cf, sf, other_sf = encoder(content), encoder(style), encoder(other)
        aligned = align_features(cf, sf)
        first = decoder(aligned, content, sf)
        second = decoder(aligned, content, other_sf)
    assert first.shape == content.shape and torch.isfinite(first).all()
    assert 0 <= first.min() <= first.max() <= 1
    assert (first - second).abs().max() > 1e-7
    before = [p.detach().clone() for p in decoder.parameters()]
    encoder_before = [p.detach().clone() for p in encoder.parameters()]
    metrics = conditioned_step(
        encoder,
        decoder,
        torch.optim.Adam(decoder.parameters(), lr=1e-4),
        content,
        style,
    )
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
    assert any(
        not torch.equal(a, b) for a, b in zip(before, decoder.parameters(), strict=True)
    )
    assert all(
        torch.equal(a, b) and b.grad is None
        for a, b in zip(encoder_before, encoder.parameters(), strict=True)
    )
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in decoder.parameters()
    )
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in decoder.condition.parameters()
    )


def test_shared_baseline_initialization_and_checkpoint_loading(tmp_path) -> None:
    """New conditioning leaves existing baseline weights and loading untouched."""
    torch.manual_seed(0)
    old = GuidedDecoder()
    torch.manual_seed(0)
    new = StyleConditionedDecoder()
    assert model_fingerprint(old) == shared_baseline_fingerprint(new)
    path = tmp_path / "old.pt"
    torch.save(old.state_dict(), path)
    restored = GuidedDecoder()
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
    assert all(
        torch.equal(a, b)
        for a, b in zip(old.parameters(), restored.parameters(), strict=True)
    )


def test_comparison_rejects_incompatible_checkpoints(tmp_path) -> None:
    """Reject incorrect architecture before reading data or creating output."""
    import pytest

    from moodshift.engine.compare_conditioned import compare

    first, second = tmp_path / "a.pt", tmp_path / "b.pt"
    torch.save({"architecture": "guided-v1"}, first)
    torch.save({"architecture": "guided-v1"}, second)
    with pytest.raises(ValueError, match="Expected guided-v1"):
        compare(first, second, tmp_path / "data", tmp_path / "result")
    assert not (tmp_path / "result").exists()


def test_comparison_uses_same_inputs_without_overwriting(tmp_path, monkeypatch) -> None:
    """Exercise evaluation on synthetic fixtures and reject a reused output path."""
    import pytest
    from PIL import Image

    import moodshift.engine.compare_conditioned as comparison
    from moodshift.engine.dataset import inspect_dataset
    from moodshift.engine.train_guided import dataset_fingerprint

    root = tmp_path / "data"
    for split in ("train", "val"):
        for role in ("content", "style"):
            directory = root / split / role
            directory.mkdir(parents=True)
            for i in range(10 if split == "train" else 2):
                Image.new(
                    "RGB",
                    (32, 35),
                    (
                        i * 11,
                        30 if split == "train" else 130,
                        40 if role == "content" else 110,
                    ),
                ).save(directory / f"{i}.png")
    encoder_path = tmp_path / "encoder.pt"
    encoder_path.write_bytes(b"fixture encoder")
    import hashlib

    encoder_hash = hashlib.sha256(encoder_path.read_bytes()).hexdigest()
    metadata = {
        "seed": 0,
        "epochs": 1,
        "image_size": 96,
        "batch_size": 1,
        "optimizer": "Adam",
        "learning_rate": 1e-4,
        "gradient_clip_norm": 1.0,
        "device": "cpu",
        "threads": 2,
        "torch_version": str(torch.__version__),
        "pairing": "fixture",
        "dataset_sha256": dataset_fingerprint(root, inspect_dataset(root).files),
        "encoder_checkpoint_sha256": encoder_hash,
        "loss_weights": {"style": 1},
        "initial_decoder_sha256": "shared",
    }
    new_metadata = {**metadata, "shared_baseline_initial_sha256": "shared"}
    first, second = tmp_path / "baseline.pt", tmp_path / "conditioned.pt"
    torch.save(
        {
            "architecture": "guided-v1",
            "epoch": 1,
            "experiment": metadata,
            "decoder": GuidedDecoder().state_dict(),
        },
        first,
    )
    torch.save(
        {
            "architecture": "style-film-v1",
            "epoch": 1,
            "experiment": new_metadata,
            "decoder": StyleConditionedDecoder().state_dict(),
        },
        second,
    )
    monkeypatch.setattr(comparison, "pretrained_checkpoint_path", lambda: encoder_path)
    monkeypatch.setattr(
        comparison,
        "load_feature_extractor",
        lambda device, allow_download: SqueezeNetFeatures(
            squeezenet1_1(weights=None).features
        ),
    )
    previous_threads = torch.get_num_threads()
    try:
        comparison.compare(first, second, root, tmp_path / "result")
        assert len(list((tmp_path / "result").glob("*.png"))) == 4
        assert len((tmp_path / "result/metrics.csv").read_text().splitlines()) == 5
        before = (tmp_path / "result/metrics.csv").read_bytes()
        with pytest.raises(FileExistsError):
            comparison.compare(first, second, root, tmp_path / "result")
        assert before == (tmp_path / "result/metrics.csv").read_bytes()
    finally:
        torch.set_num_threads(previous_threads)
