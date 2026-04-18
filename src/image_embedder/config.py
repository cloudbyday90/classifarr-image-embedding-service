# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import tomllib
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Optional TOML config file — loaded once at import time.
# Docker Compose mounts ./config.toml into /app/config.toml read-only (0444).
# Override the path via CONFIG_FILE env var (useful for tests or local dev).
# Environment variables always take precedence over config file values.
# ---------------------------------------------------------------------------
_CONFIG_PATH = os.getenv("CONFIG_FILE", "/app/config.toml")
_TOML: dict = {}

if os.path.isfile(_CONFIG_PATH):
    with open(_CONFIG_PATH, "rb") as _f:
        _TOML = tomllib.load(_f)


def _get_bool(value: str | None, default: bool) -> bool:
    """Parse a string env-var value as a boolean."""
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _c(section: str, key: str):
    """Return a value from the TOML config, or None if absent."""
    return _TOML.get(section, {}).get(key)


def _str(env_key: str, section: str, cfg_key: str, default: str) -> str:
    v = os.getenv(env_key)
    if v is not None:
        return v
    c = _c(section, cfg_key)
    return str(c) if c is not None else default


def _int(env_key: str, section: str, cfg_key: str, default: int) -> int:
    v = os.getenv(env_key)
    if v is not None:
        return int(v)
    c = _c(section, cfg_key)
    return int(c) if c is not None else default


def _bool(env_key: str, section: str, cfg_key: str, default: bool) -> bool:
    v = os.getenv(env_key)
    if v is not None:
        return _get_bool(v, default)
    c = _c(section, cfg_key)
    if c is not None:
        return bool(c)
    return default


def _get_csv_list(value: str) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass
class Settings:
    host: str = field(default_factory=lambda: _str("IMAGE_EMBEDDER_HOST", "server", "host", "0.0.0.0"))
    port: int = field(default_factory=lambda: _int("IMAGE_EMBEDDER_PORT", "server", "port", 8000))
    default_model: str = field(default_factory=lambda: _str("DEFAULT_MODEL", "model", "default_model", "ViT-L-14"))
    device: str = field(default_factory=lambda: _str("DEVICE", "model", "device", "auto"))
    allow_remote_urls: bool = field(default_factory=lambda: _bool("ALLOW_REMOTE_IMAGE_URLS", "image", "allow_remote_urls", False))
    allowed_remote_hosts: list[str] = field(
        default_factory=lambda: (
            _get_csv_list(os.getenv("ALLOWED_REMOTE_IMAGE_HOSTS", ""))
            if os.getenv("ALLOWED_REMOTE_IMAGE_HOSTS") is not None
            else (_c("image", "allowed_remote_hosts") or [])
        )
    )
    max_image_bytes: int = field(default_factory=lambda: _int("MAX_IMAGE_BYTES", "image", "max_image_bytes", 10485760))
    request_timeout_seconds: int = field(default_factory=lambda: _int("REQUEST_TIMEOUT_SECONDS", "image", "request_timeout_seconds", 15))

    embed_concurrency: int = field(default_factory=lambda: _int("IMAGE_EMBEDDER_CONCURRENCY", "queue", "concurrency", 1))
    embed_max_queue: int = field(default_factory=lambda: _int("IMAGE_EMBEDDER_MAX_QUEUE", "queue", "max_queue", 100))
    embed_max_wait_seconds: int = field(default_factory=lambda: _int("IMAGE_EMBEDDER_MAX_WAIT_SECONDS", "queue", "max_wait_seconds", 60))
    embed_batch_window_ms: int = field(default_factory=lambda: _int("EMBED_BATCH_WINDOW_MS", "queue", "batch_window_ms", 0))
    embed_batch_max_size: int = field(default_factory=lambda: _int("EMBED_BATCH_MAX_SIZE", "queue", "batch_max_size", 8))
    embed_batch_api_max_items: int = field(default_factory=lambda: _int("EMBED_BATCH_API_MAX_ITEMS", "queue", "batch_api_max_items", 32))
    embed_cache_size: int = field(default_factory=lambda: _int("EMBED_CACHE_SIZE", "model", "embed_cache_size", 1000))
    warmup_on_startup: bool = field(default_factory=lambda: _bool("WARMUP_ON_STARTUP", "model", "warmup_on_startup", True))

    log_level: str = field(default_factory=lambda: _str("LOG_LEVEL", "logging", "level", "INFO"))
    log_file: str | None = field(default_factory=lambda: os.getenv("LOG_FILE") or _c("logging", "file") or None)
    log_max_bytes: int = field(default_factory=lambda: _int("LOG_MAX_BYTES", "logging", "max_bytes", 10 * 1024 * 1024))
    log_backup_count: int = field(default_factory=lambda: _int("LOG_BACKUP_COUNT", "logging", "backup_count", 5))
    log_json_format: bool = field(default_factory=lambda: _bool("LOG_JSON_FORMAT", "logging", "json_format", False))

    memory_cleanup_interval_seconds: int = field(
        default_factory=lambda: _int("MEMORY_CLEANUP_INTERVAL_SECONDS", "memory", "cleanup_interval_seconds", 300)
    )
    max_process_memory_mb: int | None = field(
        default_factory=lambda: _int("MAX_PROCESS_MEMORY_MB", "memory", "max_process_memory_mb", 0) or None
    )
    max_gpu_memory_mb: int | None = field(
        default_factory=lambda: _int("MAX_GPU_MEMORY_MB", "memory", "max_gpu_memory_mb", 0) or None
    )
    cleanup_on_shutdown: bool = field(default_factory=lambda: _bool("CLEANUP_ON_SHUTDOWN", "memory", "cleanup_on_shutdown", True))
    embed_cleanup_every_n: int = field(
        default_factory=lambda: _int("EMBED_CLEANUP_EVERY_N", "memory", "embed_cleanup_every_n", 0)
    )

    shutdown_timeout_seconds: int = field(
        default_factory=lambda: _int("SHUTDOWN_TIMEOUT_SECONDS", "server", "shutdown_timeout_seconds", 30)
    )

    # Authentication
    # SERVICE_API_KEY is intentionally not in config.toml — it must stay out of source control.
    # Set it via .env (generated by scripts/generate_env.py).
    service_api_key: str | None = field(default_factory=lambda: os.getenv("SERVICE_API_KEY") or None)
    require_api_key: bool = field(default_factory=lambda: _bool("REQUIRE_API_KEY", "auth", "require_api_key", True))
    rate_limit_embed: str = field(default_factory=lambda: _str("RATE_LIMIT_EMBED", "auth", "rate_limit_embed", "30/minute"))
    rate_limit_health: str = field(default_factory=lambda: _str("RATE_LIMIT_HEALTH", "auth", "rate_limit_health", "120/minute"))
