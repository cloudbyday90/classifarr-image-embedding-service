# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Run inside a built image: native probe, non-root cache, HTTP startup/shutdown."""

import argparse
import json
import os
import secrets
import signal
import subprocess
import sys
import time
from pathlib import Path
from tempfile import TemporaryFile
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from backend_probe import probe_backend


def fetch_json(url: str, api_key: str | None = None) -> dict | list:
    request = Request(url, headers={"X-Api-Key": api_key} if api_key else {})
    with urlopen(request, timeout=2) as response:
        return json.load(response)


def check_service(port: int, device: str) -> None:
    api_key = secrets.token_hex(32)
    env = dict(os.environ)
    env.update(
        DEVICE=device,
        WARMUP_ON_STARTUP="false",
        CLEANUP_ON_SHUTDOWN="false",
        REQUIRE_API_KEY="true",
        SERVICE_API_KEY=api_key,
        CONFIG_FILE="/nonexistent/backend-smoke.toml",
        LOG_FILE="",
    )
    base_url = f"http://127.0.0.1:{port}"
    with TemporaryFile(mode="w+b") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "image_embedder.main:app",
             "--host", "127.0.0.1", "--port", str(port)],
            env=env, stdout=log, stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 60
            while True:
                if process.poll() is not None:
                    raise RuntimeError("Service exited before health became available")
                try:
                    health = fetch_json(f"{base_url}/health")
                    break
                except (URLError, TimeoutError):
                    if time.monotonic() >= deadline:
                        raise RuntimeError("Service startup exceeded 60 seconds")
                    time.sleep(0.1)
            if not isinstance(health, dict) or health.get("status") != "ok":
                raise RuntimeError(f"Unexpected health response: {health}")
            device_info = health.get("device")
            if not isinstance(device_info, dict) or device_info.get("type") != device.split(":")[0]:
                raise RuntimeError(f"Service did not select {device}: {device_info}")
            ready = fetch_json(f"{base_url}/ready")
            if not isinstance(ready, dict) or ready.get("ready") is not False:
                raise RuntimeError(f"Unloaded model unexpectedly ready: {ready}")
            try:
                fetch_json(f"{base_url}/models")
            except HTTPError as exc:
                if exc.code != 401:
                    raise RuntimeError(f"Unexpected auth refusal: {exc.code}") from exc
            else:
                raise RuntimeError("Protected endpoint accepted a missing API key")
            fetch_json(f"{base_url}/models", api_key)
        except Exception:
            log.seek(0)
            sys.stderr.write(log.read().decode("utf-8", errors="replace"))
            raise
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
                raise RuntimeError("Service did not stop within 10 seconds")
        # Uvicorn can re-raise SIGTERM after completing its lifespan teardown.
        log.seek(0)
        stopped_cleanly = b"Application shutdown complete." in log.read()
        if process.returncode not in (0, -signal.SIGTERM) or not stopped_cleanly:
            raise RuntimeError(f"Service shutdown failed: exit {process.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=("cpu", "cuda", "cuda-legacy", "openvino"))
    parser.add_argument("--require-gpu", action="store_true", help="Fail unless CUDA inference executes")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    getuid = getattr(os, "getuid", None)
    if getuid is None or getuid() == 0:
        raise RuntimeError("Container smoke must run as the shipped non-root user")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    cache = Path(os.environ.get("OV_MODEL_CACHE") or os.environ["HF_HOME"])
    result = probe_backend(args.backend, cache, require_gpu=args.require_gpu)
    device = "openvino:CPU" if args.backend == "openvino" else "cuda" if result["gpu_inference"] else "cpu"
    check_service(args.port, device)
    result.update(non_root=True, cache_writable=True, service_device=device, service_startup=True, clean_shutdown=True)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
