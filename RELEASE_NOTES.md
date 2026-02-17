# Release Notes

## Unreleased

### Highlights
- N/A

### What's New
- N/A

### Improvements
- N/A

### Fixes
- N/A

## v0.0.1.2-alpha

### Highlights
- Production-ready robustness: structured logging, memory management, graceful shutdown, and Kubernetes-ready health probes.

### What's New
- `GET /ready` endpoint for Kubernetes readiness probes
- `POST /admin/cleanup` endpoint for manual memory cleanup
- Structured JSON logging support via `LOG_JSON_FORMAT=true`
- Configurable log rotation (`LOG_MAX_BYTES`, `LOG_BACKUP_COUNT`)
- Model warmup on startup via `WARMUP_ON_STARTUP=true`

### Improvements
- Enhanced `/health` endpoint with device, model status, memory info, and queue stats
- Automatic periodic memory cleanup (GPU cache + garbage collection)
- Graceful shutdown with configurable timeout and cleanup
- Global exception handler for unhandled errors

### Fixes
- N/A

## v0.0.1.1-alpha

### Highlights
- Safer remote image fetching (streaming size cap + SSRF hardening; remote URLs now disabled by default).
- Expanded offline test suite and API error-path coverage.

### What's New
- New env var `ALLOWED_REMOTE_IMAGE_HOSTS` to allowlist image hosts when `ALLOW_REMOTE_IMAGE_URLS=true`.

### Improvements
- More explicit request validation for `image_size`.
- Concurrency-safe model loading and clearer device selection errors.

### Fixes
- Enforce `MAX_IMAGE_BYTES` while streaming remote image downloads (prevents oversized images from buffering in memory).

## v0.0.1.0-alpha

### Highlights
- ?? Initial image embedding service release

### What's New
- `/health`, `/models`, and `/embed-image` endpoints
- Docker image build and container runtime support

### Improvements
- N/A

### Fixes
- N/A
