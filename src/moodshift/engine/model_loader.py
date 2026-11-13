"""Lazy official SqueezeNet loading through the normal PyTorch weight cache."""

import hashlib
import pickle
from pathlib import Path
from urllib.parse import urlparse

import torch
from torchvision.models import SqueezeNet1_1_Weights, squeezenet1_1

from moodshift.engine.exceptions import DeviceError, ModelLoadError
from moodshift.engine.features import SqueezeNetFeatures


def select_device(requested: str = "auto") -> torch.device:
    """Use available CUDA or CPU; honor explicit requests or raise an error."""
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise DeviceError("CUDA was requested but is not available; select cpu or auto")
    return torch.device(
        "cuda" if requested != "cpu" and torch.cuda.is_available() else "cpu"
    )


def pretrained_checkpoint_path() -> Path:
    """Locate the official checkpoint in torch.hub's normal cache (TORCH_HOME aware)."""
    filename = Path(urlparse(SqueezeNet1_1_Weights.DEFAULT.url).path).name
    return Path(torch.hub.get_dir()) / "checkpoints" / filename


def load_feature_extractor(
    device: torch.device, *, allow_download: bool = True
) -> SqueezeNetFeatures:
    """Load only official ImageNet weights, verifying cached hash before reuse.

    TorchVision verifies downloads. A corrupt cache is reported rather than
    deleted or replaced automatically. Reuse an engine instance to avoid reloads.
    """
    checkpoint = pretrained_checkpoint_path()
    try:
        if checkpoint.exists():
            expected = checkpoint.stem.rsplit("-", 1)[1]
            with checkpoint.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if not digest.startswith(expected):
                raise ModelLoadError(
                    "Cached SqueezeNet weights failed checksum verification; "
                    "remove the corrupt SqueezeNet checkpoint manually and retry"
                )
        elif not allow_download:
            raise ModelLoadError(
                "Official SqueezeNet weights are not cached; download disabled"
            )
        # Preserve the caller's random stream while TorchVision constructs modules.
        with torch.random.fork_rng(devices=[]):
            model = squeezenet1_1(weights=SqueezeNet1_1_Weights.DEFAULT, progress=False)
        extractor = SqueezeNetFeatures(model.features)
    except (
        OSError,
        RuntimeError,
        MemoryError,
        ValueError,
        EOFError,
        pickle.UnpicklingError,
    ) as error:
        if isinstance(error, ModelLoadError):
            raise
        raise ModelLoadError(
            "Unable to load official SqueezeNet weights; "
            "check network, cache, and packages"
        ) from error
    try:
        return extractor.to(device).eval()
    except (RuntimeError, MemoryError) as error:
        raise DeviceError(
            "Cannot place SqueezeNet on selected device; try CPU or smaller images"
        ) from error
