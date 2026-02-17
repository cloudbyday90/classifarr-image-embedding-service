# Classifarr Image Embedding Service

A lightweight image embedding microservice designed for Classifarr. It exposes a small HTTP API to generate image embeddings from poster URLs or base64 payloads, and to list supported models.

## Features
- FastAPI service with `GET /health`, `GET /ready`, `GET /models`, `POST /embed-image`
- CLIP-based image embeddings (ViT-L/14 and ViT-B/16)
- Optional L2 normalization
- Docker-ready with simple configuration
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
      - DEFAULT_MODEL=ViT-L-14
      - DEVICE=auto
      - ALLOW_REMOTE_IMAGE_URLS=true
      - REQUEST_TIMEOUT_SECONDS=15
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
    restart: unless-stopped
```

In this example, Classifarr can reach the image embedder at:
- Host: `image-embedder` (same compose network)
- Port: `8000`

## GPU Examples
These examples expose GPU devices to the container. Whether the service actually uses them depends on your PyTorch build and drivers.

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

### Intel iGPU (VAAPI / OpenVINO)
```yaml
services:
  image-embedder:
    image: ghcr.io/cloudbyday90/classifarr-image-embedder:latest
    ports:
      - "8000:8000"
    devices:
      - /dev/dri:/dev/dri
    group_add:
      - "video"
```

Note: Intel iGPU acceleration requires compatible drivers and an image that supports that runtime. This base image is CPU/CUDA focused; use iGPU-specific builds if needed.

## Local Development (No Docker)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
uvicorn image_embedder.main:app --host 0.0.0.0 --port 8000
```

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

## Environment Variables

### Core Settings
- `IMAGE_EMBEDDER_PORT` (default `8000`)
- `IMAGE_EMBEDDER_HOST` (default `0.0.0.0`)
- `DEFAULT_MODEL` (default `ViT-L-14`)
- `DEVICE` (default `auto` -> cuda if available, else cpu)
- `ALLOW_REMOTE_IMAGE_URLS` (default `false`)
- `ALLOWED_REMOTE_IMAGE_HOSTS` (comma-separated host allowlist)
- `MAX_IMAGE_BYTES` (default `10485760` - 10MB)
- `REQUEST_TIMEOUT_SECONDS` (default `15`)

### Concurrency & Queue
- `IMAGE_EMBEDDER_CONCURRENCY` (default `1`)
- `IMAGE_EMBEDDER_MAX_QUEUE` (default `100`)
- `IMAGE_EMBEDDER_MAX_WAIT_SECONDS` (default `60`)

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
- `SHUTDOWN_TIMEOUT_SECONDS` (default `30` - max time for graceful shutdown)

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

## License
Classifarr Image Embedding Service is licensed under GPL-3.0 (or later). See `LICENSE`.

## Copyright Compliance
All source files should include current copyright headers consistent with Classifarr standards.

Check compliance:
```bash
python scripts/check_copyright.py
```
