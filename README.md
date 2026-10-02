# Classifarr Image Embedding Service

A lightweight image embedding microservice designed for Classifarr. It exposes a small HTTP API to generate image embeddings from poster URLs or base64 payloads, and to list supported models.

## Features
- FastAPI service with `GET /health`, `GET /ready`, `GET /models`, `POST /embed-image`, `POST /embed-batch`
- CLIP-based image embeddings (ViT-L/14 and ViT-B/16)
- Optional L2 normalization
- Docker-ready with simple configuration
- **CUDA GPU build** (`Dockerfile.cuda` + `docker-compose.cuda.yml`) for NVIDIA GPU inference
- **API key authentication** with constant-time comparison (service-to-service)
- **Per-key rate limiting** on `/embed-image` to prevent GPU resource exhaustion
- **Production-ready robustness:**
  - Structured logging with rotation
  - Automatic memory management (GPU + garbage collection)
  - Graceful shutdown with cleanup
  - Global error handling
  - Kubernetes-ready health probes

## Quickstart (Docker)
Build and run locally:

```bash
docker build -t classifarr-image-embedder:local .
docker run --rm -p 8000:8000 classifarr-image-embedder:local
```

Default bind address is `0.0.0.0:8000` inside the container. The default port Classifarr should use is `8000`.

## Classifarr Configuration (Host + Port)
Classifarr needs to reach this service from the Classifarr server container.

Use one of the following patterns based on your setup:

- Same Docker Compose project:
  - Host: `image-embedder` (service name)
  - Port: `8000`
- Separate containers on the same Docker host (Docker Desktop Windows/Mac):
  - Host: `host.docker.internal`
  - Port: `8000`
- Separate containers on the same Docker host (Linux):
  - Add `extra_hosts: ["host.docker.internal:host-gateway"]` to the Classifarr service
  - Host: `host.docker.internal`
  - Port: `8000`
- Running the service on the host (not in Docker):
  - Host: `localhost`
  - Port: `8000`

In the Classifarr UI: **Settings -> RAG & Embeddings -> Image Embeddings**, set the host and port above, then **Test Connection** and **Save**.

## Docker Compose (Recommended)

```yaml
services:
  image-embedder:
    image: ghcr.io/cloudbyday90/classifarr-image-embedder:latest
    container_name: image-embedder
    ports:
      - "8000:8000"
    environment:
      - SERVICE_API_KEY=${SERVICE_API_KEY}   # shared with Classifarr; set via .env
    configs:
      - source: app_config
        target: /app/config.toml
        mode: 0444
    restart: unless-stopped

  classifarr:
    image: ghcr.io/cloudbyday90/classifarr:latest
    container_name: classifarr
    ports:
      - "21324:21324"
    extra_hosts:
      - "host.docker.internal:host-gateway"
    environment:
      - TZ=America/New_York
      - IMAGE_EMBEDDER_API_KEY=${SERVICE_API_KEY}   # same key, sent as X-Api-Key header
    restart: unless-stopped

configs:
  app_config:
    file: ./config.toml
```

In this example, Classifarr can reach the image embedder at:
- Host: `image-embedder` (same compose network)
- Port: `8000`

## Authentication

The embedding service uses a **shared API key** for service-to-service authentication. Classifarr generates and manages this key; you copy it into your environment once.

### Setup

The easiest way is the start script — it auto-generates `.env` on first run, prompts you to copy the key into Classifarr, then starts the stack.

**Windows (PowerShell):**
```powershell
.\scripts\start.ps1
```

**Linux / macOS:**
```bash
bash scripts/start.sh
```

The script does three things:
1. If `.env` doesn't exist yet, runs `generate_env.py` to create it with a fresh `SERVICE_API_KEY`.
2. Prints the key and pauses so you can copy it into Classifarr before the service starts.
3. Runs `docker compose up -d`.

**`.env` persists across restarts.** Once it exists, `docker compose up -d` (or the start script) just works — no key generation or prompts. You only go through the first-time flow once per machine.

---

**Manual setup / key rotation:**

```bash
python scripts/generate_env.py           # first-time: creates .env + config.toml defaults
python scripts/generate_env.py --force   # rotate key (config.toml is preserved)
```

This creates:
- `.env` — contains only `SERVICE_API_KEY`. Gitignored; never committed.
- `config.toml` — default settings, if not already present. Committed to the repo, mounted read-only into the container.

Copy the printed `SERVICE_API_KEY` value into Classifarr as `IMAGE_EMBEDDER_API_KEY` (env var), or via Classifarr Settings → API Keys with the `embed_service` tier.

Then start the stack:
```bash
docker compose up -d
```

### How it works

- Classifarr sends the key as an `X-Api-Key` header (or `Authorization: Bearer <key>`) on every request.
- The embedding service validates it with a constant-time comparison (`hmac.compare_digest`) to prevent timing attacks.
- `/health` and `/ready` are **always public** — Docker and orchestrators need these unauthenticated.
- `/admin/cleanup` is **always protected**, even in development mode.

### Modes

| `require_api_key` in `config.toml` | `SERVICE_API_KEY` | Behaviour |
|---|---|---|
| `true` (default) | set | All endpoints except `/health`/`/ready` require the key |
| `true` | _not set_ | Service returns `503` — fail-closed on misconfiguration |
| `false` | set | Dev mode — non-admin endpoints are public; `/admin/cleanup` still requires key |
| `false` | _not set_ | Dev mode — non-admin endpoints public; admin returns `503` |

> **Tip:** set `require_api_key = false` in `config.toml` only for local development where the service is not network-accessible. You can also override it temporarily with `REQUIRE_API_KEY=false` as an environment variable.

## GPU / CUDA Build

A dedicated `Dockerfile.cuda` ships **Torch 2.14.1 CUDA 13.0 wheels** for **linux/amd64** on a digest-pinned `nvidia/cuda:13.0.3-base-ubuntu24.04` base. Torch's wheel dependencies provide CUDA libraries and cuDNN. The default supports Turing and newer GPUs, including Blackwell; Maxwell, Pascal, and Volta require the explicit legacy profile below.

Use a compatible driver from branch 580 or newer; Linux 580.126.20 or newer is the corresponding CUDA 13.0.3 toolkit driver. Windows/WSL drivers are installed separately. See the [backend design and compatibility record](docs/backend-build-recommendation.md) for official sources and architecture limits.

**Prerequisites:** [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/) installed on the Docker host.

**Build and run (Compose override — recommended):**
```bash
docker compose -f docker-compose.yml -f docker-compose.cuda.yml up -d
```
The override sets `DEVICE=cuda`, adds the GPU device reservation, and points the build at `Dockerfile.cuda`. All other settings (API key, config mount, model cache volume) are inherited from `docker-compose.yml`.

**Build standalone:**
```bash
docker build -f Dockerfile.cuda -t classifarr-image-embedder:cuda .
docker run --rm --gpus all -p 8000:8000 -e DEVICE=cuda classifarr-image-embedder:cuda
```

**Verify GPU is in use:**
```bash
curl http://localhost:8000/health | python -m json.tool
# "device": {"type": "cuda", "name": "NVIDIA GeForce RTX ..."}
```

> The default `Dockerfile` / `docker-compose.yml` remain CPU-only. `DEVICE=auto` in `config.toml` will fall back to CPU automatically if CUDA is unavailable at runtime.

**Legacy amd64 GPUs (CUDA 12.6):** select the base, profile, and index together. This profile excludes Blackwell. Prefer Linux driver 560.35.05 or Windows driver 561.17 or newer compatible versions.

```bash
docker build -f Dockerfile.cuda -t classifarr-image-embedder:cuda-legacy \
  --build-arg CUDA_BASE=nvidia/cuda:12.6.3-base-ubuntu24.04@sha256:c87e78933f4c16e3272123bf2f75537306596d0fbaa395a29696a22786e5ee0e \
  --build-arg TORCH_PROFILE=requirements-torch-cuda-legacy.txt \
  --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu126 .
```

---

## AMD ROCm Build

> **Windows is not supported.** Setting `DEVICE=rocm` on a Windows host raises an error at startup. ROCm requires a Linux host. Windows ROCm support is tracked upstream and will be re-evaluated when it reaches production status.

A dedicated `Dockerfile.rocm` ships PyTorch with **ROCm 6.4 wheels** targeting AMD discrete GPUs and Ryzen APUs on Linux. Because ROCm exposes the same `torch.cuda.*` API as CUDA, the entire inference code path is shared — no code changes are needed relative to the CUDA build; only the base image and wheel index differ.

**Supported hardware (Linux only):**
- AMD Instinct MI300X / MI350X and other datacenter GPUs (fully supported)
- AMD Radeon RX 7000 (RDNA3) and RX 9000 (RDNA4) discrete GPUs
- AMD Ryzen AI Max / Ryzen AI 300 iGPU (preview support via ROCm 6.4.4+)

**Host prerequisites (Linux):**
- ROCm 6.4+ installed on the host *or* use the official `rocm/pytorch` Docker image
- Add your user to the `render` and `video` groups: `sudo usermod -aG render,video $USER`
- Verify ROCm sees the GPU: `rocm-smi`

**`DEVICE` setting:** Use `DEVICE=rocm` in `config.toml` or as an environment variable. On Linux with a visible AMD GPU this resolves to `torch.device("cuda")` internally. Setting it on Windows raises:
```
ValueError: ROCm (AMD GPU) is not supported on Windows. ROCm requires a Linux host.
```

**Verify ROCm is in use** (returns `"type": "rocm"` instead of `"type": "cuda"` when torch was built with HIP):
```bash
curl http://localhost:8000/health | python -m json.tool
# "device": {"type": "rocm", "hip_version": "6.4.43482-...", "name": "AMD Radeon RX 7900 XTX"}
```

> ROCm on Windows via WSL2 reached beta in ROCm 6.1 but is **not yet production-ready** as of April 2026. This service will add a `Dockerfile.rocm` and Compose override once Windows ROCm stabilises.

---

## Intel OpenVINO Build

A dedicated `Dockerfile.openvino` targets Intel hardware via **OpenVINO 2026.4.0**, with matching Python bindings and a digest-pinned `openvino/ubuntu24_runtime:2026.4.0` base. This image supports **linux/amd64**. CPU-only Torch 2.14.1 handles initial export.

**Supported hardware (all via a single `DEVICE=openvino:AUTO`):**
- Intel integrated GPU: 12th-gen Core (UHD 770) and newer, including Core Ultra series
- Intel Arc discrete GPU: A-series, B-series (Battlemage), and future Arc generations
- Intel CPU: all modern Intel CPUs via the OpenVINO CPU plugin (AVX-512 / VNNI optimized)

**How it works:** On first startup the service exports the HuggingFace CLIP model to OpenVINO IR format (`.xml` + `.bin`) and caches it in the model volume. Subsequent restarts load the cached IR directly — no re-export, no PyTorch on the hot path.

**Host prerequisites (Linux):**
- Intel GPU kernel driver (`xe` for 12th-gen+, or `i915`); kernel 6.2+ recommended
- DRI device nodes present: `ls /dev/dri/renderD*`
- Add your user to the `render` and `video` groups: `sudo usermod -aG render,video $USER`
- Export host group IDs (recommended to avoid `/dev/dri` permission mismatches):
  - `export VIDEO_GID="$(getent group video | cut -d: -f3)"`
  - `export RENDER_GID="$(stat -c '%g' /dev/dri/renderD128)"`

**Model conversion:** We use `openvino.convert_model()` directly to preserve CLIP `image_embeds` output.

**Build and run (Compose override — recommended):**
```bash
docker compose -f docker-compose.yml -f docker-compose.openvino.yml up -d
```
The override sets `DEVICE=openvino:AUTO`, mounts `/dev/dri`, adds the `render` and `video` groups, and points the build at `Dockerfile.openvino`.

**Build standalone:**
```bash
docker build -f Dockerfile.openvino -t classifarr-image-embedder:openvino .
docker run --rm --device /dev/dri:/dev/dri --group-add "${VIDEO_GID:-44}" --group-add "${RENDER_GID:-109}" \
  -p 8000:8000 -e DEVICE=openvino:AUTO classifarr-image-embedder:openvino
```

**Force a specific device:**
```bash
# CPU-only (no GPU required):
DEVICE=openvino:CPU

# Intel GPU only (iGPU or Arc):
DEVICE=openvino:GPU

# Specific discrete GPU when multiple are present:
DEVICE=openvino:GPU.0
```

**Verify OpenVINO is in use:**
```bash
curl http://localhost:8000/health | python -m json.tool
# "device": {"type": "openvino", "ov_device": "AUTO", "available_devices": ["CPU", "GPU"]}
```

**WSL2 (Windows):** Replace the `devices` block with `/dev/dxg` and mount `/usr/lib/wsl`. See the comments in `docker-compose.openvino.yml` for details.

---

## GPU Examples
These examples expose GPU devices to the container using the pre-built image. Whether the service actually uses them depends on your PyTorch build — use the `Dockerfile.cuda` build above if you need GPU inference.

### NVIDIA (CUDA / NVENC)
```yaml
services:
  image-embedder:
    image: ghcr.io/cloudbyday90/classifarr-image-embedder:latest
    ports:
      - "8000:8000"
    environment:
      - DEVICE=cuda
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

If your Docker setup ignores `deploy`, use `device_requests` instead:
```yaml
    device_requests:
      - driver: nvidia
        count: all
        capabilities: [gpu]
```

### Intel iGPU / Arc GPU (VAAPI / OpenVINO)
```yaml
services:
  image-embedder:
    image: ghcr.io/cloudbyday90/classifarr-image-embedder:openvino
    ports:
      - "8000:8000"
    environment:
      - DEVICE=openvino:AUTO
    devices:
      - /dev/dri:/dev/dri
    device_cgroup_rules:
      - "c 226:* rmw"
    group_add:
      - "${VIDEO_GID:-44}"
      - "${RENDER_GID:-109}"
```

Use `DEVICE=openvino:AUTO` and the OpenVINO build (see [Intel OpenVINO Build](#intel-openvino-build)) for hardware-accelerated inference on Intel GPUs and CPUs.

### AMD GPU (ROCm) — Linux only

> **Not supported on Windows.** See [AMD ROCm Build](#amd-rocm-build) for details.

```yaml
services:
  image-embedder:
    image: ghcr.io/cloudbyday90/classifarr-image-embedder:rocm
    ports:
      - "8000:8000"
    environment:
      - DEVICE=rocm
    devices:
      - /dev/kfd:/dev/kfd
      - /dev/dri:/dev/dri
    group_add:
      - video
      - render
```

## Local Development (No Docker)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt -r requirements-dev.txt torch==2.14.1
$env:PYTHONPATH = "src"
uvicorn image_embedder.main:app --host 0.0.0.0 --port 8000
```

On Linux/macOS, activate with `source .venv/bin/activate` and set `export PYTHONPATH=src` before running the same install/server commands.

## Offline Container Validation

PRs and default-branch pushes build and smoke CPU on amd64 and arm64, plus modern CUDA, legacy CUDA, and OpenVINO on amd64, without publishing images. Each check audits the actual installed packages through OSV and runs the native libraries, tiny CLIP projection, non-root cache access, and service startup/authentication/shutdown without model downloads. OpenVINO also exports and reloads IR. Release publishing remains tag-only and depends on these checks.

CUDA ARM is deferred: the selected upstream cuSPARSELt 0.8.1 aarch64 wheel contains an incompatible internal SBSA platform tag and fails `pip check`. The build retains this failure; see the [backend design and outcome](docs/backend-build-recommendation.md) for the exact evidence and restoration criteria.

```bash
docker build -t classifarr-image-embedder:cpu-smoke .
docker run --rm --network none --cap-drop ALL --security-opt no-new-privileges \
  --entrypoint python classifarr-image-embedder:cpu-smoke scripts/smoke_backend.py --backend cpu
```

Use `--backend cuda`, `cuda-legacy`, or `openvino` with the corresponding image. To require real NVIDIA GPU execution, add `--gpus all` to Docker and `--require-gpu` to the script. Without GPU access, CUDA smoke verifies its wheel profile and CPU projection. OpenVINO smoke uses its CPU plugin; Intel GPU deployments need separate hardware validation. The script verifies readiness remains false without production weights.

## API
### GET /health
Returns service health with device, model, and memory info.

Response body:
```json
{
  "status": "ok",
  "provider": "local",
  "default_model": "ViT-L-14",
  "device": {"type": "cuda", "name": "NVIDIA GeForce RTX 3080"},
  "models": [
    {"name": "ViT-L-14", "loaded": true},
    {"name": "ViT-B-16", "loaded": false}
  ],
  "memory": {"allocated_mb": 1024.5, "reserved_mb": 2048.0},
  "queue": {"concurrency": 1, "in_flight": 0, "waiting": 0, ...}
}
```

### GET /ready
Returns readiness status for Kubernetes-style probes. Returns 200 when the default model is loaded.

Response body:
```json
{
  "ready": true,
  "default_model_loaded": true,
  "device": {"type": "cuda", "name": "NVIDIA GeForce RTX 3080"}
}
```

### GET /models
Returns supported models and metadata.

### POST /embed-image
Request body:
```json
{
  "image_url": "https://example.com/poster.jpg",
  "model": "ViT-L-14",
  "normalize": true,
  "image_size": 512
}
```

Response body:
```json
{
  "embedding": [0.1, 0.2, 0.3],
  "dims": 768,
  "provider": "local",
  "model": "ViT-L-14",
  "image_size": 512
}
```

### POST /admin/cleanup
Trigger manual memory cleanup (garbage collection + GPU cache clearing).

Response body:
```json
{
  "gc_collected": 150,
  "gpu_freed_mb": 256.5,
  "process_rss_mb": 1024.0,
  "gpu_allocated_mb": 512.0,
  "gpu_reserved_mb": 768.0
}
```

### Request Deadlines and Shutdown

Embedding requests return HTTP 504 when `REQUEST_TIMEOUT_SECONDS` expires. A worker already running continues to hold its concurrency slot until it finishes, because Python cannot forcibly stop an inference thread. `/health` and `X-Queue-In-Flight` include this work after the response has been sent. With `IMAGE_EMBEDDER_MAX_QUEUE=0`, retries receive HTTP 429 while every slot is occupied. Requests that expire or are canceled while waiting for admission are removed before inference starts.

Single requests, `/embed-batch`, and batch-window dispatch share the same execution service. Shutdown closes admission (HTTP 503 for new work) and drains dispatched inference for up to `SHUTDOWN_TIMEOUT_SECONDS`. If the drain budget expires, the service logs the remaining work and skips shutdown memory cleanup. Uvicorn retains control of process signals. Configure Uvicorn's graceful-shutdown timeout and the container/process supervisor's stop timeout to allow request draining plus this application drain budget; a supervisor must enforce any hard stop for a stuck native inference thread.

The optional batch window uses the same bounded admission budget as `/embed-batch`. It reserves one computation slot before collecting compatible requests, so `in_flight` includes collection and shared-lock acquisition as well as running inference. Waiting batch members each count toward `IMAGE_EMBEDDER_MAX_QUEUE`; health and queue headers include them. New groups fail with HTTP 429 when the waiting room is full. With a zero waiting limit, compatible requests can still join an admitted collecting group up to `EMBED_BATCH_MAX_SIZE`, but another group cannot wait for capacity.

Queue deadlines start at admission, before collection; a queued group's members inherit its first member's deadline. Once admitted, the collection window starts and cannot be extended by later joins. Cancellation removes pending membership, and dispatch rebuilds the payload from live clients after shared-lock acquisition. With concurrency C, batch maximum B, and waiting maximum Q, admitted single-image jobs and explicit-batch requests are bounded by C × B + Q. An explicit-batch request can contain multiple images; request-count limits work alongside the input budgets below.

### Input Size and Pixel Budgets

HTTP bodies have a 16 MiB ceiling, counting received bytes even without Content-Length. Oversized bodies return HTTP 413 before JSON parsing completes. Known oversized inline images/batches are refused before queue admission. Starlette's header-based body refusal may use plain text; image refusals use JSON `detail`. Successful response schemas are unchanged.

Each compressed image remains limited to 10 MiB. Source and projected CLIP pre-crop resize each allow at most 16,000,000 pixels; this also refuses extreme aspect ratios that would create a large resize intermediate. A batch resolves at most 32 MiB of compressed image data, including cache hits, and retains uncached images within a 32,000,000-pixel budget counting source plus projected resize. Items refused during worker processing remain ordered per-item batch errors; a coalesced single request receives HTTP 413 for its own refused image. Decoded images close after work or errors, including when the HTTP caller has already timed out.

All ceilings must be positive integers. Configure them in `[image]` or the environment variables below. The default raw body ceiling permits one near-10-MiB base64 image, but large inline batches may need smaller images or a deliberately raised ceiling. These limits do not impose an exact RSS quota: size model/tensor memory and configure server/proxy concurrency and container memory limits, especially with multiple workers. See the [input budget design and outcome](docs/input-memory-budgets.md).

## Environment Variables

### Core Settings
- `IMAGE_EMBEDDER_PORT` (default `8000`)
- `IMAGE_EMBEDDER_HOST` (default `0.0.0.0`)
- `DEFAULT_MODEL` (default `ViT-L-14`)
- `DEVICE` (default `auto` -> cuda if available, else cpu)
- `ALLOW_REMOTE_IMAGE_URLS` (default `false`)
- `ALLOWED_REMOTE_IMAGE_HOSTS` (comma-separated host allowlist)
- `MAX_IMAGE_BYTES` (default `10485760` - 10 MiB compressed image)
- `MAX_REQUEST_BODY_BYTES` (default `16777216` - 16 MiB raw HTTP body)
- `MAX_IMAGE_PIXELS` (default `16000000` - source and pre-crop resize ceilings)
- `MAX_BATCH_IMAGE_BYTES` (default `33554432` - 32 MiB aggregate compressed inputs)
- `MAX_BATCH_IMAGE_PIXELS` (default `32000000` - aggregate source + pre-crop resize pixels)
- `REQUEST_TIMEOUT_SECONDS` (default `15`)

### Concurrency & Queue
- `IMAGE_EMBEDDER_CONCURRENCY` (default `1`)
- `IMAGE_EMBEDDER_MAX_QUEUE` (default `100`)
- `IMAGE_EMBEDDER_MAX_WAIT_SECONDS` (default `60`)
- `EMBED_BATCH_WINDOW_MS` (default `0` - disables coalescing; positive values collect compatible admitted requests)
- `EMBED_BATCH_MAX_SIZE` (default `8` - maximum members per coalesced computation)
- `EMBED_BATCH_API_MAX_ITEMS` (default `32` - maximum images per `/embed-batch` request)

### Startup
- `WARMUP_ON_STARTUP` (default `true` - preload default model)

### Logging
- `LOG_LEVEL` (default `INFO` - DEBUG, INFO, WARNING, ERROR)
- `LOG_FILE` (path to log file, optional)
- `LOG_MAX_BYTES` (default `10485760` - 10MB before rotation)
- `LOG_BACKUP_COUNT` (default `5` - number of rotated files)
- `LOG_JSON_FORMAT` (default `false` - use JSON log format)

### Memory Management
- `MEMORY_CLEANUP_INTERVAL_SECONDS` (default `300` - periodic cleanup)
- `MAX_PROCESS_MEMORY_MB` (threshold for health check, 0 to disable)
- `MAX_GPU_MEMORY_MB` (threshold for health check, 0 to disable)
- `CLEANUP_ON_SHUTDOWN` (default `true` - cleanup on graceful shutdown)

### Graceful Shutdown
- `SHUTDOWN_TIMEOUT_SECONDS` (default `30` - application inference drain budget after server shutdown begins)

### Authentication
- `SERVICE_API_KEY` — shared secret validated on every protected request; must match the key configured in Classifarr Settings. Set via `.env`, never in `config.toml`.

Auth mode and rate limiting are configured in `config.toml` under `[auth]` and can be overridden by environment variable:
- `REQUIRE_API_KEY` — overrides `config.toml` `require_api_key` (default `true`)
- `RATE_LIMIT_EMBED` — overrides `config.toml` `rate_limit_embed` (default `30/minute`); format: `<count>/<period>` e.g. `10/minute`, `2/second`

### Configuration File
Most settings are defined in `config.toml` (committed to the repo) and mounted read-only into the container at `/app/config.toml`. Environment variables always override config file values. Override the path with `CONFIG_FILE=/path/to/config.toml`.

## Release Workflow
- Keep release notes in `RELEASE_NOTES.md` with an `Unreleased` section at the top (high-level, user-facing, emojis/graphs ok).
- Keep technical changes in `CHANGELOG.md`.
- When shipping a release, move `Unreleased` into a versioned section (e.g., `v0.1.1`).
- Bump the version in `src/image_embedder/__init__.py` and `src/image_embedder/main.py`.
- Follow `release.md` for the full checklist and tag creation.

## Tests
```bash
pytest
```

## Engineering Decisions

- [Input body and image allocation design and validation](docs/input-memory-budgets.md)
- [Open PR 33 local implementation and validation](docs/pr-33-local-validation.md)
- [Inference execution design and validation](docs/inference-execution.md)
- [Bounded batch admission design and validation](docs/bounded-batch-admission.md)
- [Local PR 28 implementation and validation](docs/pr-28-local-validation.md)
- [Local PR 42 implementation and validation](docs/pr-42-local-validation.md)
- [Backend build design and validation](docs/backend-build-recommendation.md)
- [Local PR 40 implementation and validation](docs/pr-40-local-validation.md)
- [Recommendation stack and next task](docs/recommendation-stack.md)

## License
Classifarr Image Embedding Service is licensed under GPL-3.0 (or later). See `LICENSE`.

## Copyright Compliance
All source files should include current copyright headers consistent with Classifarr standards.

Check compliance:
```bash
python scripts/check_copyright.py
```
