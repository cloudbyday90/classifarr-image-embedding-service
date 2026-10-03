# Changelog

All notable technical changes to this project are documented in this file.
Release notes (`RELEASE_NOTES.md`) are high-level and user-facing.

## Unreleased

### Added
- Real HTTP/1 upload-capacity calibration at body/image ceilings, with observed ingress rejection, disconnect recovery and native CPU/OpenVINO vector checks.
- A focused socket-capacity reference for the repository calibration AI skill, with separate design, results and PR-availability records.
- A focused repository AI skill for dependency migrations, with native graph review, compatibility boundaries and separate design/outcome records.
- Recurring native Windows validation for private setup, remote-child/shared-batch cleanup and HTTP response lifetimes, with complete platform locks, mandatory outcome gates and separate graph audits.
- A focused repository AI skill for native platform validation, with separate research, design and outcome records.
- Concurrent inline/remote and detached-owner capacity probes with task/thread, anonymous temporary-storage and CUDA allocator measurements.
- A focused repository AI skill for deployment capacity calibration, with separate research, design and outcome records.
- Shared total remote batch fetch budgets with ordered partial results, retained child ownership and native Linux/Windows regressions.
- A repository AI skill for designing and validating embedding-service resource and ownership changes.
- Configurable total HTTP response-send lifetimes, with modular server/transport services, native slow-reader/disconnect regressions and separate design/outcome records.
- Configurable total upload and remote-fetch lifetimes, with modular ASGI deadline, disposable-worker protocol and supervision services; separate design/outcome records and native cancellation/network regressions.
- Modular private setup/publication and Windows ACL helpers, with separate design/outcome records and cross-platform setup regressions.
- Reviewed native CI tool contracts, modular verification and release-retention helpers, with separate design/outcome records and workflow boundary regressions.
- Complete hash-locked Linux dependency profiles, modular generation/installation validation, and separate dependency, refresh-policy and PR 50 design/outcome records.
- Modular offline capacity probes with process/cgroup memory, batch parity, authenticated deadline and detached-owner checks; a manual CPU/OpenVINO calibration workflow and separate design/outcome records.
- Modular pinned model catalog, verified source loading and atomic versioned OpenVINO artifact services, with separate model/PR 41 design records.
- Offline publisher preprocessing and cache-integrity regressions, plus opt-in production-weight probes and change-scoped/weekly CPU/OpenVINO validation CI.
- Modular application-wide HTTP ingress admission, shared configured server/healthcheck entrypoints, and separate ingress/PR 51 design and validation records.
- Modular remote URL/transport services, destination/TLS regressions, and separate security, PR 49, and Python/Rust design records.
- Configurable HTTP body, image pixel, and aggregate batch input budgets, with pre-admission inline checks and allocation/resource-lifetime regressions.
- Offline backend build/smoke CI for CPU, modern CUDA, legacy CUDA, and OpenVINO, including tiny CLIP projection and OpenVINO IR export/reload checks.
- Backend compatibility, dependency-selection, and local PR 40 design/validation records.
- Shared FIFO admission service and regressions for bounded batch retention, waiting telemetry, and live-client dispatch.
- Application-scoped inference execution service shared by single, bulk, and coalesced embedding paths.
- Event-controlled regression tests and design records for execution ownership, shutdown, and local PR validation.

### Changed
- Include socket uploads in manual CPU/OpenVINO capacity calibration; retain existing production limits and document authentication before body receipt as the next fix.
- Migrated API test clients to HTTPX2 with explicit lifespan/state cleanup checks and a gate against the deprecated Starlette fallback; production dependency graphs remain unchanged.
- Raised the declared Transformers minimum to 5.17.0 through local PR 53 adoption while retaining the reviewed 5.18.0 runtime.
- Raise the NumPy minimum to the reviewed 2.5.3 version; preserve complete locked wheel graphs.
- Extend manual CPU/OpenVINO calibration with concurrent mixed inputs and bounded temporary storage/task experiments.
- Adopt open PR #46's renewed FastAPI minimum of 0.142.2 while retaining the reviewed wheel graphs.
- Adopted PR 27's Trivy action update through a verified immutable v0.36.0 commit and explicit native scanner path, preserving disabled setup/cache paths and existing scan/report gates.
- The shared launcher uses asyncio so response completion accounts for TLS ciphertext buffers before releasing capacity.
- Adopt open PR #52 locally by raising the pytest development minimum to the already hash-locked 9.1.1 release.
- Adopt open PR #34 locally by raising the HTTPX development minimum to the already hash-locked 0.28.1 release.
- Keep generated API keys out of setup/startup console output by default; provide explicit new-key display and rotation controls while preserving existing configuration.
- Freeze remaining workflow actions and nested scanner/builder dependencies; narrow credentials, remove Gitleaks PR-write access and require completed OSV scans.
- Replace Docker Hub cleanup shell interpolation with bounded authenticated API handling, complete inventory validation and current-tag protection.
- Adopt open PR #54 locally by raising the Requests minimum to the already hash-locked 2.34.2 release.
- Adopt open PR #50 locally by raising the Uvicorn minimum to the tested 0.54.0 release.
- Install reviewed binary wheels in Docker and Linux QA; run weekly validation with isolated, hash-locked audit tooling and exact runtime inventories.
- Separate the embedding HTTP deadline from the remote-hop timeout: default to 45 seconds for embeddings while preserving explicit legacy overrides and the 15-second remote default.
- Apply open PR #45 locally by raising the Pydantic minimum to the tested stable 2.13.5 release.
- Raise the OpenVINO Compose default to a tunable 8 GiB/no-swap ceiling after measured large-model cold export exceeded 4 GiB; CPU/CUDA retain their existing default.
- Load supported CLIP models from immutable publisher snapshots using explicit image-only PIL preprocessing; compile saved FP32 OpenVINO IR with the accuracy execution hint on first use and reload.
- Apply open PR #41 locally by updating the external Gitleaks action to verified Node 24 v3.0.0, pinned to its immutable commit.
- Explicitly configure shipped worker, server concurrency and backlog budgets; apply tunable Compose memory/no-swap ceilings across backend profiles.
- Apply open PR #51 locally by updating the OSV reusable PR workflow to verified 2.6.0, pinned to its immutable commit.
- Apply open PR #49 locally by upgrading setup-python to verified ESM v7.0.0, pinned to an immutable commit in both Python workflows.
- Require Starlette's supported streaming body limiter and apply open PR #33 locally by raising the pytest-cov minimum to 7.1.0.
- Container builds use exact backend-specific Torch profiles, constrained shared dependencies, and verified base-image digests; CPU and OpenVINO use CPU-only wheels.
- Default CUDA builds support CUDA 13.0 and Blackwell, with an explicit CUDA 12.6 legacy profile; OpenVINO bindings and base now match 2026.4.0.
- Tag-only publishing waits for backend validation; weekly Docker base updates complement existing Dependabot checks.
- Validated CUDA support is amd64; ARM CUDA remains blocked by upstream cuSPARSELt wheel metadata, with dependency checks kept strict.
- Optional batch collection reserves capacity before retaining jobs; waiting members share the global queue limit with explicit batch requests.
- Timed-out inference remains visible in queue telemetry until its worker finishes; expired queue admissions are removed.

### Fixed
- Serialize cold model initialization across aliases and owners within each process after reproducing a concurrent Transformers import failure; cached models retain their direct return path.
- Remove inherited OpenVINO Python-environment files before copying the verified builder environment; native smoke checks reject duplicate package metadata.
- Reject image expansion before RGB/CLIP preprocessing and close decoded images on success, failures, and completed detached inference.
- Repaired the missing CUDA base and OpenVINO's collision with the Intel base's existing non-root identity.
- Batch-window bursts can no longer bypass queue limits, and expired members are removed from payloads before worker dispatch.
- Request deadlines and caller cancellation no longer release capacity while inference threads continue running.
- Graceful shutdown preserves Uvicorn's signal handlers, drains inference work, and skips memory cleanup when active work exceeds the drain budget.
- Batch-window shutdown settles collected job futures and retains accounting for running inference.

### Security
- Bound remote DNS, TLS, redirects and trickled responses with one parent-owned fetch budget; strip ambient credentials and terminate/reap expired workers before releasing inference capacity.
- Expire stalled/trickled uploads with HTTP 408 while retaining ingress ownership through downstream cleanup and response completion.
- Retain default secret-scanning rules with exact path/value exceptions for two verified historical log digests, validated against negative native scanner controls.
- Publish complete `.env` data with owner-only permissions established before writing; refuse linked targets, serialize setup/rotation and preserve existing keys on handled publication failures.
- Reject stale dependency inputs, unapproved wheel origins, altered artifacts and unexpected installed packages; require hashes for all transitive Python dependencies without index fallback or source builds.
- Verify publisher asset digests before model loading; restrict PyTorch weight deserialization and remote code, and rebuild corrupted or incomplete generated artifacts under finite process locks.
- Reject excess complete HTTP requests before body receive/JSON retention with retryable 503 responses, while preserving health access and detached inference ownership.
- Adopt OSV's report-completion guard and locally verify missing-report failure and native differential vulnerability reporting.
- Validate every remote-image redirect and connect only to approved public numeric addresses, preserving verified TLS identity, bounded streaming, and existing download-error status contracts.
- Refuse ambiguous/credentialed URLs, unsafe DNS answers and HTTPS downgrades; isolate remote fetching from ambient proxy/netrc/CA overrides and sanitize download failures.
- Refuse oversized streamed bodies and inline payloads before queue retention; bound batch compressed data and source/resize pixel work without weakening Pillow's global safety policy.
- Raised Pillow, Transformers, Torch, and packaged setuptools minimums to published patched versions; strict OSV audits cover the installed environment and each validated runtime image.
- Applied open PR #40 locally by upgrading the direct OSV scanner to 2.3.8 and pinning its actual container digest.
- Applied open PR #42 locally by upgrading checkout to verified v7.0.1, pinned to an immutable commit across six workflows.
- Raised the development pytest minimum to 9.0.3, applying open PR #28 locally and including its upstream temporary-directory security fix.

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
