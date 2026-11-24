"""Testable UI-to-service integration, request guards, and temporary PNG downloads."""

import logging
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from PIL import Image

from moodshift.config import DEFAULT_CONFIG, AppConfig
from moodshift.imaging.io import export_image, load_image
from moodshift.imaging.reference_mood import reference_mood_boost
from moodshift.services.mood_transfer import MoodTransferService
from moodshift.services.references import resolve_preset
from moodshift.ui.helpers import format_metadata
from moodshift.ui.state import RenderedResult, UiParameters

type ServiceFactory = Callable[[bool, str], MoodTransferService]


class UiController:
    """Serialize reusable backends; isolate submissions and files by browser session."""

    def __init__(
        self,
        *,
        config: AppConfig = DEFAULT_CONFIG,
        service_factory: ServiceFactory | None = None,
    ) -> None:
        """Create an owned temporary export directory; never load weights at startup."""
        self.config = config
        self._factory = service_factory or self._create_service
        self._services: dict[tuple[bool, str], MoodTransferService] = {}
        self._temporary = TemporaryDirectory(prefix="moodshift-ui-")
        self.output_dir = Path(self._temporary.name)
        self._sessions: set[str] = set()
        self._session_lock = threading.Lock()
        self._processing_lock = threading.Lock()

    def _create_service(self, offline: bool, device: str) -> MoodTransferService:
        return MoodTransferService(
            config=self.config, device=device, allow_download=not offline
        )

    @contextmanager
    def claim(self, session: str) -> Iterator[None]:
        """Reject duplicate work for the same session and always release the guard."""
        with self._session_lock:
            if session in self._sessions:
                raise ValueError(
                    "This session is already generating. Wait for it to finish."
                )
            self._sessions.add(session)
        try:
            yield
        finally:
            with self._session_lock:
                self._sessions.discard(session)

    def preview(self, path: str | None) -> Image.Image | None:
        """Decode the upload through existing size/format/dimension validation."""
        return load_image(path, self.config) if path else None

    def generate(
        self, parameters: UiParameters, *, stage: Callable[[str], None]
    ) -> RenderedResult:
        """Call the service and export safely; report stages without fake steps."""
        stage("Validating inputs")
        if not parameters.content:
            raise ValueError("Upload a content image before generating.")
        settings = parameters.settings()
        preset, reference = parameters.selection()
        resolve_preset(preset, reference, settings.mode)
        original = load_image(parameters.content, self.config)
        with self._processing_lock:
            key = (parameters.offline, parameters.device)
            if key not in self._services:
                self._services[key] = self._factory(*key)
            if settings.intensity == 0 or (
                settings.mode == "neural" and settings.neural_style_strength == 0
            ):
                stage("Returning unchanged normalized content")
            elif settings.mode == "grade" or settings.neural_style_strength == 0:
                stage("Applying photographic grade")
            else:
                stage("Running neural transfer — CPU optimization may take time")
            result = self._services[key].transfer(
                parameters.content,
                preset_id=preset,
                reference=reference,
                settings=settings,
            )
        image = result.image
        boost_applied = bool(
            parameters.boost_enabled
            and parameters.boost_reference
            and parameters.boost_strength > 0
            and result.metadata.neural_applied
        )
        if boost_applied:
            stage("Applying reference photographic color transfer")
            with load_image(parameters.boost_reference, self.config) as boost_reference:
                image = reference_mood_boost(
                    image, boost_reference, parameters.boost_strength
                )
        stage("Preparing comparison and PNG download")
        self._prune_exports()
        download = export_image(image, self.output_dir / f"moodshift-{uuid4().hex}.png")
        before = (
            original
            if original.size == image.size
            else original.resize(image.size, Image.Resampling.LANCZOS)
        )
        summary, metadata = format_metadata(result.metadata)
        metadata["reference_mood_boost"] = {
            "applied": boost_applied,
            "strength": parameters.boost_strength if boost_applied else 0,
            "reference_source": "user" if boost_applied else "none",
        }
        if boost_applied:
            summary += (
                "\n\n**Reference Mood Boost:** Applied photographic LAB color "
                "transfer after neural processing."
            )
        elif parameters.boost_enabled and settings.mode in ("neural", "hybrid"):
            summary += (
                "\n\n**Reference Mood Boost:** Not applied (requires an uploaded "
                "boost reference, nonzero strength, and neural processing)."
            )
        return RenderedResult(image, (before, image), download, summary, metadata)

    def _prune_exports(self) -> None:
        # Only our flat, UUID-named exports are removed, never uploads/user paths.
        cutoff = time.time() - 3600
        for path in self.output_dir.glob("moodshift-*.png"):
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
            except OSError:
                logging.getLogger(__name__).debug(
                    "A temporary export could not be expired"
                )

    def close(self) -> None:
        """Clean only this controller's owned temporary files after app shutdown."""
        self._temporary.cleanup()
