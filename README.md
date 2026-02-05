# Classifarr Image Embedding Service

A lightweight image embedding microservice designed for Classifarr. It exposes a small HTTP API to generate image embeddings from poster URLs or base64 payloads, and to list supported models.

## Features
- FastAPI service with `/health`, `/models`, and `/embed-image`
- CLIP-based image embeddings (ViT-L/14 and ViT-B/16)
- Optional L2 normalization
- Docker-ready

## Quickstart
1. Create a virtual environment.
2. Install dependencies.
3. Run the service.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
uvicorn image_embedder.main:app --host 0.0.0.0 --port 8000
```

## Docker
```bash
docker build -t classifarr-image-embedder:local .
docker run --rm -p 8000:8000 classifarr-image-embedder:local
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

## GPU Notes
For NVIDIA GPUs, install CUDA-enabled PyTorch and run with an NVIDIA runtime. See PyTorch docs for the correct wheel index for your CUDA version.

## Tests
```bash
pytest
```
