import os
from dataclasses import dataclass


def _get_bool(value: str, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass
class Settings:
    host: str = os.getenv("IMAGE_EMBEDDER_HOST", "0.0.0.0")
    port: int = int(os.getenv("IMAGE_EMBEDDER_PORT", "8000"))
    default_model: str = os.getenv("DEFAULT_MODEL", "ViT-L-14")
    device: str = os.getenv("DEVICE", "auto")
    allow_remote_urls: bool = _get_bool(os.getenv("ALLOW_REMOTE_IMAGE_URLS"), True)
    max_image_bytes: int = int(os.getenv("MAX_IMAGE_BYTES", "10485760"))
    request_timeout_seconds: int = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "15"))
