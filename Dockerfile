# ── Build stage: compile wheels and install dependencies ────────────────────
FROM ubuntu:24.04@sha256:008173c23f95b170204355c12626cb5a965d779a7e1283b09e9cffbb1bf33ca3 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-torch-cpu.txt /tmp/

RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -r /tmp/requirements-torch-cpu.txt \
        --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt /tmp/
RUN /opt/venv/bin/pip install -r /tmp/requirements.txt \
        -c /tmp/requirements-torch-cpu.txt \
    && /opt/venv/bin/pip check

# ── Runtime stage: minimal image, no build tools, non-root user ──────────────
FROM ubuntu:24.04@sha256:008173c23f95b170204355c12626cb5a965d779a7e1283b09e9cffbb1bf33ca3

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/app/src"
# Store HuggingFace model cache inside the app dir where appuser has write access
ENV HF_HOME="/app/.cache"

# Runtime-only system libraries (Pillow needs libgl1 / libglib2.0-0)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    libgl1 \
    libglib2.0-0 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY src ./src
COPY scripts/backend_probe.py scripts/smoke_backend.py scripts/production_model_probe.py ./scripts/

# CIS Docker Benchmark 4.1: do not run as root
RUN groupadd --gid 1001 appgroup \
    && useradd --uid 1001 --gid 1001 --no-create-home --shell /sbin/nologin appuser \
    && mkdir -p /app/.cache \
    && chown -R appuser:appgroup /app/.cache
USER appuser

EXPOSE 8000

# CIS Docker Benchmark 4.6: add a HEALTHCHECK
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -m image_embedder.healthcheck

CMD ["python", "-m", "image_embedder.server"]
