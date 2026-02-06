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
    # NOTE: use default_factory so env vars are read when Settings() is instantiated,
    # not at import time (important for tests and predictable runtime behavior).
    host: str = field(default_factory=lambda: os.getenv("IMAGE_EMBEDDER_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("IMAGE_EMBEDDER_PORT", "8000")))
    default_model: str = field(default_factory=lambda: os.getenv("DEFAULT_MODEL", "ViT-L-14"))
    device: str = field(default_factory=lambda: os.getenv("DEVICE", "auto"))
    # Defaults to false for SSRF safety; prefer passing base64 image bytes from Classifarr.
    allow_remote_urls: bool = field(default_factory=lambda: _get_bool(os.getenv("ALLOW_REMOTE_IMAGE_URLS"), False))
    allowed_remote_hosts: list[str] = field(
        default_factory=lambda: _get_csv_list(os.getenv("ALLOWED_REMOTE_IMAGE_HOSTS", ""))
    )
    max_image_bytes: int = field(default_factory=lambda: int(os.getenv("MAX_IMAGE_BYTES", "10485760")))
    request_timeout_seconds: int = field(default_factory=lambda: int(os.getenv("REQUEST_TIMEOUT_SECONDS", "15")))

    # Concurrency and queueing for embeddings.
    # - embed_concurrency limits concurrent embed computations to avoid contention/VRAM thrash.
    # - embed_max_queue controls how many requests are allowed to wait for a slot (0 = no waiting allowed).
    # - embed_max_wait_seconds bounds how long a request will wait for a slot before returning 504.
    embed_concurrency: int = field(default_factory=lambda: int(os.getenv("IMAGE_EMBEDDER_CONCURRENCY", "1")))
    embed_max_queue: int = field(default_factory=lambda: int(os.getenv("IMAGE_EMBEDDER_MAX_QUEUE", "100")))
    embed_max_wait_seconds: int = field(default_factory=lambda: int(os.getenv("IMAGE_EMBEDDER_MAX_WAIT_SECONDS", "60")))
