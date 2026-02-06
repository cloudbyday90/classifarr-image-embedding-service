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
