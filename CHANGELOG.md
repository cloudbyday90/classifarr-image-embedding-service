# Changelog

All notable technical changes to this project are documented in this file.
Release notes (`RELEASE_NOTES.md`) are high-level and user-facing.

## Unreleased

### Added
- N/A

### Changed
- N/A

### Fixed
- N/A

### Security
- N/A

## v0.0.1.3-alpha — 2026-04-18

### Added
- **P0 / P1 implementation plan** (`IMPLEMENTATION_PLAN.md`) committed to the repository as a standing reference for embedding math and invariant hardening work.
- **XOR input validation** (`EmbedImageRequest`, `EmbedBatchItem`): `@model_validator(mode="after")` enforces exactly one of `image_url` or `image_base64` per request. Dual-input and zero-input payloads are rejected with HTTP 422 before reaching the embedder.
- **Byte-hash cache keys**: cache keys are now `sha256(image_bytes) | model | image_size | normalize`. Image bytes are resolved before the cache lookup; changing content at the same URL can no longer return a stale cached embedding.
- **Immutable cache entries** (`CachedEmbedding`): `@dataclass(frozen=True, slots=True)` with `embedding: tuple[float, ...]`. Mutating a returned embedding list cannot corrupt future cache hits.
- **Embedding result validation** (`_validate_embedding_result`): rejects non-1D (nested) vectors, `len(embedding) != dims`, `dims != spec.dims`, and any non-finite value (`NaN` / `Inf` / `-Inf`). Applied in both `embed()` and `embed_batch()` before caching and returning.
- **NumPy normalization helper** (`_normalize_embedding_np`): mirrors PyTorch `F.normalize(p=2, dim=-1, eps=1e-12)` semantics for single vectors and batches. Zero vectors are epsilon-clamped (no `NaN`); non-finite input and output are rejected. Used in both OpenVINO single and batch inference paths.
- **OpenVINO output shape validation**: output shape is validated as `(1, spec.dims)` for single and `(N, spec.dims)` for batch before indexing. A wrong-shaped output now raises `ValueError` immediately.
- **`image_size` semantics frozen**: `embed()` rejects any `image_size != spec.image_size` with `ValueError` (HTTP 400 at embed route). Batch route rejects non-default values with HTTP 422. Accepted: `None` (uses spec default) or the exact spec value.
- **Canonical route response metadata**: `routes/embed.py` and `routes/batch.py` populate `model` and `image_size` response fields from the resolved `ModelSpec`, not from the embedder-returned tuple.
- **AMD ROCm device support** (`DEVICE=rocm`): on Linux with a ROCm GPU resolves to `torch.device("cuda")` internally; on Windows raises `ValueError` at startup. `get_device_info()` surfaces `{"type": "rocm", "hip_version": "..."}` on ROCm hosts. README updated with AMD ROCm Build section.
- **CUDA Docker variant** (`Dockerfile.cuda`, `docker-compose.cuda.yml`): PyTorch CUDA 12.4 wheels, `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu24.04` runtime base.
- **Intel OpenVINO Docker variant** (`Dockerfile.openvino`, `docker-compose.openvino.yml`, `requirements-openvino.txt`): exports `CLIPVisionModelWithProjection` to OpenVINO IR on first startup and caches to disk; subsequent restarts bypass PyTorch. Supports Intel iGPU 12th-gen+, Arc discrete GPUs, and AVX-512/VNNI CPU.
- **`POST /embed-batch`** endpoint: bulk image embedding up to `embed_batch_api_max_items` (default 32). Per-item `status: ok/error` results; HTTP 413 when limit exceeded.
- **LRU embedding cache** (`EMBED_CACHE_SIZE`, default 1000): both `embed()` and `embed_batch()` check cache before model inference. Hit/miss/eviction stats on `GET /health`.
- **New test files**: `tests/test_embedder_advanced.py`, `tests/test_p1_invariants.py` (185 tests total), `tests/test_test_suite_layout.py`, `tests/test_integration.py` (15 ASGI lifespan tests), `tests/test_easy_wins.py` (targeted coverage).

### Changed
- `embed()` and `embed_batch()` use byte-resolved image paths; cache key derivation is content-addressed throughout.
- Existing tests updated to correct per-model embedding dimensions (512 for ViT-B-16, 768 for ViT-L-14).
- Coverage baseline raised to 89.4% lines / 79.4% branches (from 86.3% / 70.9%).
- Switched from `CLIPModel` to `CLIPVisionModelWithProjection`, eliminating `UNEXPECTED key` warnings and reducing memory usage by ~50%.
- `request_timeout_seconds` (default 15 s) now enforced on the embedding call via `asyncio.wait_for`; hung inference returns HTTP 504.
- Refactored application wiring: `lifecycle.py`, `routes/health.py`, `routes/models.py`, `routes/admin.py`, `routes/embed.py`, `routes/batch.py`.
- Optional async batch coalescing (`embed_batch_window_ms`): requests in the same window grouped by `(model, image_size)` and dispatched in a single forward pass.
- Optional per-embed cleanup cadence (`embed_cleanup_every_n`): calls `cleanup_gpu_memory()` every N embeds (default 0 = disabled).
- Added `rate_limit_health` setting (default `120/minute`) for `/health` and `/ready` rate limiting.
- Added `pattern` constraint to `EmbedImageRequest.model` field; invalid values return HTTP 422.
- Refactored test fakes into shared `tests/fakes.py` module.
- CI: bumped `google/osv-scanner-action` 2.3.3 → 2.3.5, `github/codeql-action` v3 → v4.
- Updated Python dependencies: `fastapi` 0.136.0, `torch` 2.11.0, `transformers` 5.5.4, `pydantic` 2.13.2, `numpy` 2.4.4, `pillow` 12.2.0.

### Fixed
- Stale remote-image cache: changed content at the same URL now produces a fresh embedding.
- Cache mutation: modifying a returned embedding list no longer corrupts future cache hits for the same input.

### Security
- `request_timeout_seconds` was defined but never applied, allowing hung model inference to block indefinitely. Now enforced; returns HTTP 504.
- `/health` and `/ready` now rate-limited (120/minute per IP) to prevent DDoS amplification.
- `EmbedImageRequest.model` validated against `^[a-zA-Z0-9][a-zA-Z0-9\-_\.]*$`, preventing injection via model name.

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
