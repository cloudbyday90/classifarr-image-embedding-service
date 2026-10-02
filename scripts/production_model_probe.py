# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Explicit production-weight validation; routine tests stay small and offline."""

import argparse
import asyncio
import base64
import json
import secrets
import time
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from image_embedder.config import Settings
from image_embedder.embedder import BatchItem, ImageEmbedder
from image_embedder.model_catalog import MODEL_CATALOG
from image_embedder.model_loading import (
    load_processor,
    load_vision_model,
    verified_asset,
)


def peak_rss_bytes() -> int | None:
    try:
        import resource

        # The production probes run in Linux images; value is the process high-water mark.
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    except ImportError:
        return None


def images_and_payloads():
    images, payloads = [], []
    for h, w in [(83, 129), (128, 64)]:
        y, x = np.indices((h, w))
        pixels = np.stack(
            [(x * 7) % 256, (y * 11) % 256, ((x + y) * 3) % 256], axis=-1
        ).astype(np.uint8)
        image = Image.fromarray(pixels)
        images.append(image)
        with BytesIO() as output:
            image.save(output, format="PNG")
            payloads.append(base64.b64encode(output.getvalue()).decode("ascii"))
    return images, payloads


def probe_model(name: str, backend: str) -> dict:
    import torch

    torch.set_num_threads(1)
    spec = MODEL_CATALOG[name]
    settings = Settings(
        device="cpu" if backend == "cpu" else "openvino:CPU",
        default_model=name,
        warmup_on_startup=False,
        embed_cache_size=0,
        require_api_key=False,
        cleanup_on_shutdown=False,
    )
    embedder = ImageEmbedder(settings)
    start = time.monotonic()
    embedder.warmup(name)
    warmup_seconds = time.monotonic() - start
    images, payloads = images_and_payloads()
    try:
        single = embedder.embed(None, payloads[0], name, False, 224)
        batch = embedder.embed_batch(
            spec, 224, [BatchItem(None, p, bool(i)) for i, p in enumerate(payloads)]
        )
        assert all(isinstance(value, tuple) for value in batch), batch
        service_peak = peak_rss_bytes()
        processor = load_processor(spec)
        reference = (
            embedder._load_model(spec)[0]
            if backend == "cpu"
            else load_vision_model(spec)
        )
        with torch.inference_mode():
            direct_single = reference(
                **processor(images=images[0], return_tensors="pt")
            ).image_embeds.numpy()[0]
            direct_batch = reference(
                **processor(images=images, return_tensors="pt")
            ).image_embeds.numpy()
        np.testing.assert_allclose(single[0], direct_single, rtol=1e-4, atol=1e-4)
        expected = [
            direct_batch[0],
            direct_batch[1] / max(np.linalg.norm(direct_batch[1]), 1e-12),
        ]
        for result, vector in zip(batch, expected, strict=True):
            assert isinstance(result, tuple)
            assert result[1:] == (spec.dims, "local", name, 224)
            np.testing.assert_allclose(result[0], vector, rtol=1e-4, atol=1e-4)
        reload_seconds = None
        if backend == "openvino":
            reloaded = ImageEmbedder(settings)
            start = time.monotonic()
            reloaded.warmup(name)
            reload_seconds = time.monotonic() - start
            second = reloaded.embed(None, payloads[0], name, False, 224)
            np.testing.assert_allclose(second[0], single[0], rtol=1e-6, atol=1e-6)
        asyncio.run(check_api(embedder, settings, payloads[0], single))
        return {
            "model": name,
            "revision": spec.revision,
            "backend": backend,
            "dims": spec.dims,
            "warmup_seconds": warmup_seconds,
            "ir_reload_seconds": reload_seconds,
            "service_peak_rss_bytes_before_reference": service_peak,
            "probe_peak_rss_bytes": peak_rss_bytes(),
            "single_batch_direct_parity": True,
            "authenticated_schema_unchanged": True,
            "ir_reload_parity": backend == "openvino",
            "torch": torch.__version__,
        }
    finally:
        for image in images:
            image.close()


async def check_api(embedder, settings, payload, expected) -> None:
    import httpx

    from image_embedder.main import create_app

    # An explicit fixture key validates the protected real route without external traffic.
    settings.require_api_key = True
    settings.service_api_key = secrets.token_urlsafe(24)
    app = create_app(embedder, settings)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://local"
        ) as client:
            assert (
                await client.post("/embed-image", json={"image_base64": payload})
            ).status_code == 401
            result = await client.post(
                "/embed-image",
                headers={"X-Api-Key": settings.service_api_key},
                json={
                    "image_base64": payload,
                    "model": settings.default_model,
                    "normalize": False,
                },
            )
            assert result.status_code == 200, result.text
            data = result.json()
            assert data["dims"] == expected[1] and data["model"] == expected[3]
            np.testing.assert_allclose(
                data["embedding"], expected[0], rtol=1e-6, atol=1e-6
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["cpu", "openvino"], default="cpu")
    parser.add_argument("--model", choices=list(MODEL_CATALOG))
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.download_only:
        for name in [args.model] if args.model else MODEL_CATALOG:
            for filename, _digest in MODEL_CATALOG[name].assets:
                verified_asset(MODEL_CATALOG[name], filename)
        result = {"verified_downloads": True}
    else:
        if not args.model:
            parser.error("--model is required for inference")
        result = probe_model(args.model, args.backend)
    output = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
