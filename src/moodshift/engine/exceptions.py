"""Explicit neural engine failures; never substitute fabricated results."""


class NeuralTransferError(RuntimeError):
    """Neural transfer cannot complete safely."""


class ModelLoadError(NeuralTransferError):
    """Official weights are unavailable, corrupted, or incompatible."""


class DeviceError(NeuralTransferError):
    """A requested compute device is unavailable or failed."""
