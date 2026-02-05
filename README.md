# Classifarr Image Embedding Service

A lightweight image embedding microservice designed for Classifarr. It exposes a small HTTP API to generate image embeddings from poster URLs or base64 payloads, and to list supported models.

## Features
- FastAPI service with `GET /health`, `GET /models`, and `POST /embed-image`
- CLIP-based image embeddings (ViT-L/14 and ViT-B/16)
- Optional L2 normalization
- Docker-ready with simple configuration

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
Returns service health.

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

## Environment Variables
- `IMAGE_EMBEDDER_PORT` (default `8000`)
- `IMAGE_EMBEDDER_HOST` (default `0.0.0.0`)
- `DEFAULT_MODEL` (default `ViT-L-14`)
- `DEVICE` (default `auto` -> cuda if available, else cpu)
- `ALLOW_REMOTE_IMAGE_URLS` (default `true`)
- `MAX_IMAGE_BYTES` (default `10485760`)
- `REQUEST_TIMEOUT_SECONDS` (default `15`)

## Tests
```bash
pytest
```
