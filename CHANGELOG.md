# Changelog

All notable technical changes to this project are documented in this file.
Release notes (`RELEASE_NOTES.md`) are high-level and user-facing.

## Unreleased

### Added
- Added `"rocm"` as a valid `DEVICE` setting for AMD GPU inference via ROCm/HIP. On Linux with a ROCm-capable AMD GPU visible, `DEVICE=rocm` resolves to `torch.device("cuda")` internally — the entire PyTorch inference code path is shared with the CUDA build; only the base image and wheel index differ. On Windows, setting `DEVICE=rocm` raises `ValueError` immediately at startup with an explicit "not supported on Windows" message; Windows ROCm support is tracked upstream and the guard will be removed once it reaches production status. `sys.platform` is checked before any torch import so there is no import-time side effect. `config.toml` `device` comment block updated to document `"rocm"` with the Linux-only caveat. Three new tests cover: Windows raises on startup, Linux without a visible GPU raises, Linux with a visible GPU returns `torch.device("cuda")`.
- `get_device_info()` now distinguishes ROCm from NVIDIA CUDA at runtime: when `torch.version.hip` is not `None` (i.e. the running torch was compiled with HIP), the returned `"type"` field is overridden from `"cuda"` to `"rocm"` and a `"hip_version"` field is added. The `/health` response therefore surfaces `{"type": "rocm", "hip_version": "6.4.x", "name": "AMD Radeon ..."}` on ROCm hosts.
- README: added `## AMD ROCm Build` section (supported hardware, host prerequisites, `DEVICE=rocm` behaviour, Windows unsupported callout, `/health` verification snippet, WSL2 beta status note) and an `### AMD GPU (ROCm) — Linux only` GPU Examples compose snippet with `/dev/kfd` + `/dev/dri` device mounts.

### Changed
- Switched from `CLIPModel` to `CLIPVisionModelWithProjection` in `embedder.py`, loading only the vision encoder and projection head. Eliminates `UNEXPECTED key` warnings on model load and reduces memory usage by ~50%.
- `request_timeout_seconds` (default 15s, configurable via `REQUEST_TIMEOUT_SECONDS` env var or `config.toml`) is now enforced on the embedding call in `main.py` via `asyncio.wait_for`. Requests that exceed the timeout return HTTP 504 instead of hanging indefinitely.
- Refactored application wiring: extracted FastAPI lifespan management into `lifecycle.py` and split endpoint definitions into `routes/health.py`, `routes/models.py`, `routes/admin.py`, and `routes/embed.py`. `main.py` now focuses on app assembly, shared middleware, and router registration.
- Added optional per-request cleanup cadence after inference: `embed_cleanup_every_n` / `EMBED_CLEANUP_EVERY_N` triggers `cleanup_gpu_memory()` every N embeds. Default is `0` (disabled), which is appropriate for CPU-only deployments.
- Added optional async batch coalescing for `POST /embed-image`: when `embed_batch_window_ms` is greater than `0`, requests arriving in the same window are grouped by `(model, image_size)` and dispatched through `embed_batch()` in a single forward pass (bounded by `embed_batch_max_size`). With the default `embed_batch_window_ms=0`, behavior remains the existing single-request path.
- CI: bumped `google/osv-scanner-action` from 2.3.3 to 2.3.5 (PRs #23, #24).
- CI: bumped `github/codeql-action` from v3 to v4 across all workflow steps (PRs #25, #26).
- Updated Python dependencies to latest compatible versions: `fastapi` 0.136.0, `uvicorn` 0.44.0, `starlette` 1.0.0, `pydantic` 2.13.2, `torch` 2.11.0, `transformers` 5.5.4, `numpy` 2.4.4, `pillow` 12.2.0, `requests` 2.33.1, `huggingface-hub` 1.11.0, `pytest` 9.0.3, `pytest-cov` 7.1.0.
- Added `asgi-lifespan>=2.1` to `requirements-dev.txt` to enable proper ASGI lifespan handling (startup/shutdown) in `httpx`-based tests via `LifespanManager`.
- Added `rate_limit_health` setting (default `120/minute`, configurable via `RATE_LIMIT_HEALTH` env var or `config.toml [auth]`) controlling the rate limit applied to `/health` and `/ready`.
- Added `pattern` constraint to `EmbedImageRequest.model` field — only alphanumeric characters, hyphens, underscores, and dots are accepted; invalid values are rejected with HTTP 422 before reaching the embedder.
- Refactored test suite: extracted `FakeEmbedder`, `_no_auth_settings()`, and `_png_bytes()` into a shared `tests/fakes.py` module, eliminating ~175 lines of duplication across six test files. `FakeEmbedderWithWarmup` is now a slim subclass of `FakeEmbedder`; `FakeEmbedderWithCleanup` (identical to `FakeEmbedder`) was removed entirely.
- Added `embed_cleanup_every_n` setting (default `0` = disabled, configurable via `EMBED_CLEANUP_EVERY_N` env var or `config.toml [memory]`). When set to a positive integer N, `cleanup_gpu_memory()` is called every N completed embeds to opportunistically release fragmented VRAM. Disabled by default; recommended for GPU deployments with sustained load.
- Expanded targeted coverage tests in `tests/test_easy_wins.py` for `RWLock`/`EmbedQueue` constructor guardrails, lifecycle batch-window start/stop and shutdown cleanup error handling, memory threshold breach detection and `gc.collect(2)` cadence, config `_get_bool` variants (`y`/`on`) and fallback helpers, `get_device_info` exception fallback, queue response headers on successful embed, and `BatchWindow._dispatch` exception fan-out to pending futures.
- Added integration test suite (`tests/test_integration.py`, 15 tests) covering the full ASGI lifespan. Tests exercise auth enforcement, Bearer-token acceptance, rate limiting, Pydantic model-name validation, `asyncio.TimeoutError` → 504, queue-full → 429, unhandled embedder exception → 500, unconditional admin protection, model warmup on startup, concurrent request handling, and base64 image path — all with real startup/shutdown via `LifespanManager`. Pushed `routes/embed.py` to 100% and `security.py` to 96%.
- Added in-memory LRU embedding cache to `ImageEmbedder`. Cache key is `sha256(url_or_base64_string) | model | image_size | normalize` — encoding every axis that affects the output. Both `embed()` and `embed_batch()` check the cache before model inference; `embed_batch()` only dispatches uncached items to the GPU and skips model loading entirely when all items are cache hits. Cache size is configurable via `EMBED_CACHE_SIZE` env var or `config.toml [model] embed_cache_size` (default 1000; set to 0 to disable). Hit/miss/eviction stats are exposed on `GET /health` under the `cache` field.
- Added `POST /embed-batch` endpoint for bulk image embedding. Accepts an array of up to `embed_batch_api_max_items` images (default 32, configurable via `EMBED_BATCH_API_MAX_ITEMS` env var or `config.toml [queue] batch_api_max_items`). All items share `model`, `normalize`, and `image_size` options. Returns HTTP 200 with per-item `status: "ok"/"error"` results — a single bad image does not fail the entire batch. Exceeding the configured limit returns HTTP 413. Auth and rate limiting follow the same rules as `POST /embed-image`. The endpoint routes through the existing queue and calls `embed_batch()` directly, so the LRU cache and per-item error isolation are both active. Added `EmbedBatchRequest`, `EmbedBatchItem`, `EmbedBatchItemResult`, and `EmbedBatchResponse` models. Tests in `tests/test_batch_endpoint.py` (15 tests).
- Added CUDA Docker image variant (`Dockerfile.cuda` + `docker-compose.cuda.yml`). The CUDA build installs PyTorch from the CUDA 12.4 wheel index (`https://download.pytorch.org/whl/cu124`) in the build stage and uses `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu24.04` as the runtime base, providing the CUDA runtime and cuDNN libraries needed for GPU inference without requiring the full CUDA toolkit inside the image. A `docker-compose.cuda.yml` override file sets `DEVICE=cuda`, adds the NVIDIA GPU device reservation, and points the build at `Dockerfile.cuda`; all other settings are inherited from `docker-compose.yml`. The default CPU `Dockerfile` and `docker-compose.yml` are unchanged. Updated README with a `GPU / CUDA Build` section covering the Compose override workflow and standalone build commands.
- Added Intel OpenVINO Docker image variant (`Dockerfile.openvino`, `docker-compose.openvino.yml`, `requirements-openvino.txt`). The OpenVINO build uses `openvino/ubuntu24_runtime:2026.1.0` as the runtime base and installs CPU-only PyTorch alongside `openvino>=2026.1.0` and `transformers>=5.3`. `optimum-intel` is intentionally excluded: `OVModelForFeatureExtraction` returns only `last_hidden_state` (not `image_embeds`) and is unsuitable for CLIP image embeddings; the `[openvino]` extras bracket syntax is also deprecated upstream. On first startup the service exports `CLIPVisionModelWithProjection` to OpenVINO IR format (`.xml` + `.bin`) via `openvino.convert_model()` and caches it to disk; subsequent restarts load the cached IR directly via `openvino.Core`, so PyTorch is not on the hot path after initial model export. The `_resolve_device()` method detects `device` values prefixed with `openvino` (e.g. `openvino`, `openvino:CPU`, `openvino:GPU`, `openvino:AUTO`) and returns an `ov:<DEVICE>` string instead of a `torch.device`; `_load_model()` dispatches to the new `_load_model_openvino()` path accordingly. `embed()` and `embed_batch()` detect the `ov:` device prefix and use a numpy inference path (`return_tensors="np"`, `compiled_model(dict(inputs))[0]`) with L2 normalization via NumPy. The Compose override mounts `/dev/dri`, adds a DRM `device_cgroup_rules` entry (`c 226:* rmw`), and uses numeric `group_add` defaults (`VIDEO_GID` / `RENDER_GID`) to avoid host/container render-group GID mismatch issues on iGPU and Arc hosts. Supports: Intel iGPU 12th-gen+ (UHD 770, Core Ultra), Intel Arc A/B/C-series discrete GPUs, and Intel CPU (AVX-512/VNNI). `config.toml` updated with full device-string documentation. README updated with an `Intel OpenVINO Build` section.

### Deprecated
- N/A

### Removed
- N/A

### Fixed
- N/A

### Security
- `request_timeout_seconds` was previously defined in config but never applied to the embedding thread call, allowing a hung model inference to block a request indefinitely. Now enforced; returns HTTP 504 on expiry.
- `/health` and `/ready` endpoints now rate-limited (default `120/minute` per client IP) to prevent use as a DDoS amplification vector.
- `EmbedImageRequest.model` now validated against `^[a-zA-Z0-9][a-zA-Z0-9\-_\.]*$` at the Pydantic layer, preventing injection via model name.

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
