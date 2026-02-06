# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import os
from dataclasses import dataclass, field


def _get_bool(value: str, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}

def _get_csv_list(value: str) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass
class Settings:
    host: str = os.getenv("IMAGE_EMBEDDER_HOST", "0.0.0.0")
    port: int = int(os.getenv("IMAGE_EMBEDDER_PORT", "8000"))
    default_model: str = os.getenv("DEFAULT_MODEL", "ViT-L-14")
    device: str = os.getenv("DEVICE", "auto")
    # Defaults to false for SSRF safety; prefer passing base64 image bytes from Classifarr.
    allow_remote_urls: bool = _get_bool(os.getenv("ALLOW_REMOTE_IMAGE_URLS"), False)
    allowed_remote_hosts: list[str] = field(
        default_factory=lambda: _get_csv_list(os.getenv("ALLOWED_REMOTE_IMAGE_HOSTS", ""))
    )
    max_image_bytes: int = int(os.getenv("MAX_IMAGE_BYTES", "10485760"))
    request_timeout_seconds: int = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "15"))
