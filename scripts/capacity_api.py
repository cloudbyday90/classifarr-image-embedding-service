# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Observe maximum-batch deadlines through the actual authenticated ASGI routes."""

import secrets
import time
from dataclasses import asdict, replace

import httpx
import numpy as np

from image_embedder.config import Settings
from image_embedder.embedder import ImageEmbedder


async def check_api_deadline(
    embedder: ImageEmbedder,
    settings: Settings,
    models: list[str],
    payload: str,
    size: int,
    references: dict,
) -> dict:
    from image_embedder.main import create_app

    fixture_key = secrets.token_urlsafe(24)
    protected = replace(settings, require_api_key=True, service_api_key=fixture_key)
    app = create_app(embedder, protected)
    largest = max(models, key=lambda name: embedder.resolve_model(name).dims)
    live_model = next((name for name in models if name != largest), largest)
    request = {
        "model": largest,
        "normalize": False,
        "items": [{"image_base64": payload}] * size,
    }
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://local"
        ) as client:
            unauthorized = await client.post("/embed-batch", json=request)
            if unauthorized.status_code != 401:
                raise AssertionError(
                    "Production batch route did not require authentication"
                )
            headers = {"X-Api-Key": fixture_key}
            start = time.monotonic()
            response = await client.post("/embed-batch", headers=headers, json=request)
            elapsed = time.monotonic() - start
            after_response = asdict(app.state.queue.stats())
            if response.status_code == 200:
                body = response.json()
                if body["total"] != size or body["succeeded"] != size or body["failed"]:
                    raise AssertionError("Batch API returned incomplete results")
                for index, row in enumerate(body["results"]):
                    if (
                        row["index"] != index
                        or row["model"] != largest
                        or row["dims"] != len(references[largest][0])
                    ):
                        raise AssertionError("Batch API metadata changed")
                    np.testing.assert_allclose(
                        row["embedding"], references[largest][0], rtol=1e-4, atol=1e-4
                    )
            elif response.status_code == 504:
                if (
                    response.headers.get("X-Queue-In-Flight") != "1"
                    or after_response["in_flight"] not in (0, 1)
                    or after_response["rw_readers"] != after_response["in_flight"]
                ):
                    raise AssertionError("Timed-out native work lost its ownership")
            else:
                raise AssertionError(
                    f"Unexpected batch API status {response.status_code}"
                )
            start = time.monotonic()
            live = await client.post(
                "/embed-image",
                headers=headers,
                json={"image_base64": payload, "model": live_model, "normalize": False},
            )
            live_elapsed = time.monotonic() - start
            if live.status_code != 200:
                raise AssertionError(
                    f"Live request failed after the batch: {live.status_code}"
                )
            np.testing.assert_allclose(
                live.json()["embedding"],
                references[live_model][0],
                rtol=1e-4,
                atol=1e-4,
            )
            request["items"] *= 2
            rejected = await client.post("/embed-batch", headers=headers, json=request)
            if rejected.status_code != 413:
                raise AssertionError("Configured batch ceiling was not enforced")
            settled = asdict(app.state.queue.stats())
            if settled["in_flight"] or settled["waiting"] or settled["rw_readers"]:
                raise AssertionError(
                    "Native owners did not settle after the live request"
                )
    return {
        "model": largest,
        "batch_size": size,
        "deadline_seconds": settings.embedding_timeout_seconds,
        "batch_http_status": response.status_code,
        "batch_http_seconds": elapsed,
        "queue_after_batch_response": after_response,
        "in_flight_header_at_batch_response": response.headers.get("X-Queue-In-Flight"),
        "live_http_status": live.status_code,
        "live_http_seconds": live_elapsed,
        "settled": settled,
        "unauthenticated_status": unauthorized.status_code,
        "oversized_status": rejected.status_code,
    }
