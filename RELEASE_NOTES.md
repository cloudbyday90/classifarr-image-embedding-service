# Release Notes

## v0.0.1.3-alpha — 2026-04-18

### Highlights
- Embedding pipeline correctness hardening: stale cache results, mutable cache poisoning, wrong-width embeddings, and non-finite values are now all detected and rejected.
- Bulk embedding (`POST /embed-batch`) and single embedding (`POST /embed-image`) inputs are now strictly validated — a request with both `image_url` and `image_base64` is rejected at the API boundary.

### What's New
- **Strict input validation**: each request must supply exactly one image source (`image_url` or `image_base64`). Dual-input or empty-input payloads return HTTP 422.
- **Content-addressed embedding cache**: cache keys are now derived from the SHA-256 hash of the actual image bytes rather than the URL string. Changing content at the same URL can no longer return a stale embedding from cache.
- **`image_size` locked to model spec**: requesting a non-default `image_size` for a CLIP checkpoint now returns HTTP 400 (single) or 422 (batch) with a clear error. Removes the previous misleading behavior where an unsupported size was silently accepted and ignored by the preprocessor.

### Improvements
- Embedding results are validated for correct width and finite values before being returned; impossible metadata (`model=ViT-L-14` with `dims=3`, `NaN` values, etc.) is now caught and surfaces as an error rather than a silently malformed response.
- OpenVINO inference output shape is validated before use; a wrong-output model export can no longer silently produce nested or misshapen embeddings.
- OpenVINO normalization now exactly matches the PyTorch `F.normalize(p=2, eps=1e-12)` path, including epsilon clamping for near-zero vectors.
- API response `model` and `image_size` fields are now sourced from the resolved model spec, not from embedder internals. A buggy embedder returning wrong metadata can no longer pollute the response.
- Cached embeddings are stored as immutable frozen dataclasses; mutating a returned embedding list no longer corrupts future cache hits.
- Test suite expanded to 185 tests (89.4% line / 79.4% branch coverage, up from 86.3% / 70.9%).

### Fixes
- Stale remote-image embeddings: repeated requests to a URL that serves different content now receive a fresh embedding.
- Cache mutation: callers who modify a returned embedding list no longer affect subsequent cache hits for the same input.

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
