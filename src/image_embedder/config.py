# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import os
from dataclasses import dataclass, field


def _get_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_csv_list(value: str) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass
class Settings:
    host: str = field(default_factory=lambda: os.getenv("IMAGE_EMBEDDER_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("IMAGE_EMBEDDER_PORT", "8000")))
    default_model: str = field(default_factory=lambda: os.getenv("DEFAULT_MODEL", "ViT-L-14"))
    device: str = field(default_factory=lambda: os.getenv("DEVICE", "auto"))
    allow_remote_urls: bool = field(default_factory=lambda: _get_bool(os.getenv("ALLOW_REMOTE_IMAGE_URLS"), False))
    allowed_remote_hosts: list[str] = field(
        default_factory=lambda: _get_csv_list(os.getenv("ALLOWED_REMOTE_IMAGE_HOSTS", ""))
    )
    max_image_bytes: int = field(default_factory=lambda: int(os.getenv("MAX_IMAGE_BYTES", "10485760")))
    request_timeout_seconds: int = field(default_factory=lambda: int(os.getenv("REQUEST_TIMEOUT_SECONDS", "15")))

    embed_concurrency: int = field(default_factory=lambda: int(os.getenv("IMAGE_EMBEDDER_CONCURRENCY", "1")))
    embed_max_queue: int = field(default_factory=lambda: int(os.getenv("IMAGE_EMBEDDER_MAX_QUEUE", "100")))
    embed_max_wait_seconds: int = field(default_factory=lambda: int(os.getenv("IMAGE_EMBEDDER_MAX_WAIT_SECONDS", "60")))
    warmup_on_startup: bool = field(default_factory=lambda: _get_bool(os.getenv("WARMUP_ON_STARTUP"), True))

    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_file: str | None = field(default_factory=lambda: os.getenv("LOG_FILE") or None)
    log_max_bytes: int = field(default_factory=lambda: int(os.getenv("LOG_MAX_BYTES", str(10 * 1024 * 1024))))
    log_backup_count: int = field(default_factory=lambda: int(os.getenv("LOG_BACKUP_COUNT", "5")))
    log_json_format: bool = field(default_factory=lambda: _get_bool(os.getenv("LOG_JSON_FORMAT"), False))

    memory_cleanup_interval_seconds: int = field(
        default_factory=lambda: int(os.getenv("MEMORY_CLEANUP_INTERVAL_SECONDS", "300"))
    )
    max_process_memory_mb: int | None = field(
        default_factory=lambda: int(os.getenv("MAX_PROCESS_MEMORY_MB", "0")) or None
    )
    max_gpu_memory_mb: int | None = field(
        default_factory=lambda: int(os.getenv("MAX_GPU_MEMORY_MB", "0")) or None
    )
    cleanup_on_shutdown: bool = field(default_factory=lambda: _get_bool(os.getenv("CLEANUP_ON_SHUTDOWN"), True))

    shutdown_timeout_seconds: int = field(
        default_factory=lambda: int(os.getenv("SHUTDOWN_TIMEOUT_SECONDS", "30"))
    )
