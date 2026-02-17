# Changelog

All notable technical changes to this project are documented in this file.
Release notes (`RELEASE_NOTES.md`) are high-level and user-facing.

## Unreleased

### Added
- N/A

### Changed
- N/A

### Deprecated
- N/A

### Removed
- N/A

### Fixed
- N/A

### Security
- N/A

## v0.0.1.2-alpha

### Added
- `logging_config.py` module with structured logging, JSON formatter, and file rotation support.
- `memory.py` module for GPU memory management, garbage collection, and memory health checks.
- `GET /ready` endpoint returning `ReadyResponse` with readiness status and device info.
- `POST /admin/cleanup` endpoint returning `CleanupResponse` with GC and GPU cleanup stats.
- Enhanced `/health` endpoint returning `HealthResponse` with device, models, memory, and queue stats.
- Model warmup on startup via `WARMUP_ON_STARTUP` setting (default `true`).
- Background memory cleanup loop running at `MEMORY_CLEANUP_INTERVAL_SECONDS`.
- Graceful shutdown with signal handlers (`SIGTERM`, `SIGINT`) and configurable `SHUTDOWN_TIMEOUT_SECONDS`.
- Global exception handler returning JSON 500 responses for unhandled errors.
- New response models: `ReadyResponse`, `DeviceInfo`, `ModelStatus`, `MemoryInfo`, `CleanupResponse`.

### Changed
- `/health` response expanded with `device`, `models`, `memory`, and `queue` fields.
- Startup now logs service version via `__version__` from `__init__.py`.

### Deprecated
- N/A

### Removed
- N/A

### Fixed
- N/A

### Security
- N/A

## v0.0.1.1-alpha

### Added
- Offline test suite expanded to fully cover security, validation, device selection, model caching, and embed flow (mocked).
- Allowlist support for remote image URLs via `ALLOWED_REMOTE_IMAGE_HOSTS`.

### Changed
- `ALLOW_REMOTE_IMAGE_URLS` now defaults to `false` (SSRF safety).
- `image_size` request validation enforces positive integers at the API layer.
- Model loading is guarded by per-model locks to avoid concurrent double-loads.

### Fixed
- Remote image downloads enforce `MAX_IMAGE_BYTES` while streaming (prevents buffering oversized images).

### Security
- Remote URL validation blocks non-http(s) schemes and private/reserved IP ranges (SSRF hardening).

## v0.0.1.0-alpha

### Added
- Initial FastAPI service with `/health`, `/models`, and `/embed-image`.
- CLIP model support (ViT-L/14, ViT-B/16).
- Docker build and runtime.
