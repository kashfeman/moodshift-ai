"""Validated settings independent of UI and image processing libraries."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Application defaults and limits for future input validation.

    Byte limits apply to encoded uploads; pixel limits apply to decoded images.
    All limits are inclusive. These settings do not perform image validation.
    """

    max_upload_bytes: int = 10 * 1024 * 1024
    max_image_width: int = 4096
    max_image_height: int = 4096
    max_image_pixels: int = 16_000_000
    default_preset_id: str = "cinematic"
    default_strength: float = 0.5
    output_format: str = "PNG"

    def __post_init__(self) -> None:
        """Reject invalid limits and unsupported default settings."""
        for name in (
            "max_upload_bytes",
            "max_image_width",
            "max_image_height",
            "max_image_pixels",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

        if not isinstance(self.default_preset_id, str) or not (
            self.default_preset_id and self.default_preset_id.strip()
        ):
            raise ValueError("default_preset_id must be a non-empty string")
        if (
            isinstance(self.default_strength, bool)
            or not isinstance(self.default_strength, (int, float))
            or not 0.0 <= self.default_strength <= 1.0
        ):
            raise ValueError("default_strength must be between 0 and 1 inclusive")
        if self.output_format not in ("PNG", "JPEG"):
            raise ValueError("output_format must be PNG or JPEG")


DEFAULT_CONFIG = AppConfig()
