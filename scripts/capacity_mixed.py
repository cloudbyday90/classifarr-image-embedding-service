# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Concurrent protected batches and a canceled remote owner, with real children."""

import asyncio
import secrets
import time
from dataclasses import asdict, replace
from unittest.mock import patch

import httpx
import numpy as np
from capacity_remote_fixture import remote_fixture


async def wait_for(predicate) -> None:
    async with asyncio.timeout(10):
        while not predicate():
            await asyncio.sleep(0.01)


def validate_batch(
    response, name: str, items: list[dict], references: dict, normalize: bool = True
) -> None:
    if response.status_code != 200:
        raise AssertionError(
            f"Mixed capacity request failed: HTTP {response.status_code}"
        )
    body = response.json()
    if (
        body["total"] != len(items)
        or body["succeeded"] != len(items)
        or len(body["results"]) != len(items)
        or body["failed"]
    ):
        raise AssertionError("Mixed capacity request lost ordered results")
    for index, row in enumerate(body["results"]):
        expected = references[name][index % 2]
        if normalize:
            expected = expected / max(float(np.linalg.norm(expected)), 1e-12)
        if (
            row["index"] != index
            or row["model"] != name
            or row["dims"] != len(expected)
        ):
            raise AssertionError("Mixed capacity metadata changed")
        np.testing.assert_allclose(row["embedding"], expected, rtol=1e-4, atol=1e-4)


async def check_mixed_capacity(workload) -> dict:
    from image_embedder.main import create_app

    settings = workload.embedder.settings
    size = settings.embed_batch_api_max_items
    # Two large remote bodies stay inside the shared byte budget with inline images.
    body_bytes = min(settings.max_image_bytes, settings.max_batch_image_bytes // 4)
    if size < 2:
        raise ValueError("Mixed capacity calibration requires at least two batch items")
    models = workload.models
    largest = max(models, key=lambda name: workload.embedder.resolve_model(name).dims)
    live_model = next((name for name in models if name != largest), largest)
    inline = [{"image_base64": workload.payloads[i % 2]} for i in range(size)]
    mixed = [
        {"image_url": f"http://capacity.example/{i}"} if i < 2 else item
        for i, item in enumerate(inline)
    ]
    fixture_key = secrets.token_urlsafe(24)
    protected = replace(settings, require_api_key=True, service_api_key=fixture_key)
    app = create_app(workload.embedder, protected)
    tasks = []
    completed_batches, native_errors = [], []
    original_batch = workload.embedder.embed_batch

    def checked_batch(spec, target_size, items):
        try:
            results = original_batch(spec, target_size, items)
            if len(results) != len(items):
                raise AssertionError("Native mixed batch result count changed")
            for index, (item, result) in enumerate(zip(items, results)):
                if not isinstance(result, tuple) or result[1:] != (
                    spec.dims,
                    "local",
                    spec.name,
                    target_size,
                ):
                    raise AssertionError("Native mixed batch metadata changed")
                expected = workload.references[spec.name][index % 2]
                if item.normalize:
                    expected = expected / max(float(np.linalg.norm(expected)), 1e-12)
                np.testing.assert_allclose(result[0], expected, rtol=1e-4, atol=1e-4)
            completed_batches.append(spec.name)
            return results
        except Exception as error:
            native_errors.append(type(error).__name__)
            raise

    with (
        remote_fixture(workload.payloads, body_bytes) as (started, children),
        patch.object(workload.embedder, "embed_batch", side_effect=checked_batch),
    ):
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://local"
            ) as client:
                headers = {"X-Api-Key": fixture_key}

                async def submit(name, items, normalize):
                    return await client.post(
                        "/embed-batch",
                        headers=headers,
                        json={"model": name, "items": items, "normalize": normalize},
                    )

                try:
                    unauthorized = await client.post(
                        "/embed-batch", json={"items": mixed}
                    )
                    if unauthorized.status_code != 401:
                        raise AssertionError(
                            "Mixed route did not enforce authentication"
                        )
                    began = time.monotonic()
                    first = asyncio.create_task(submit(largest, mixed, False))
                    tasks.append(first)
                    await wait_for(started.is_set)
                    second = asyncio.create_task(submit(live_model, inline, True))
                    tasks.append(second)
                    await wait_for(lambda: app.state.queue.stats().waiting == 1)
                    queued = asdict(app.state.queue.stats())
                    responses = await asyncio.gather(first, second)
                    for response, name, items, normalize in zip(
                        responses, (largest, live_model), (mixed, inline), (False, True)
                    ):
                        validate_batch(
                            response, name, items, workload.references, normalize
                        )
                    success_seconds = time.monotonic() - began
                    started.clear()
                    detached = asyncio.create_task(submit(largest, mixed, False))
                    tasks.append(detached)
                    await wait_for(started.is_set)
                    detached.cancel()
                    try:
                        await detached
                    except asyncio.CancelledError:
                        pass
                    retained = asdict(app.state.queue.stats())
                    if retained["in_flight"] != 1 or retained["rw_readers"] != 1:
                        raise AssertionError(
                            "Canceled mixed caller lost its native owner"
                        )
                    follower = asyncio.create_task(submit(live_model, inline, True))
                    tasks.append(follower)
                    await wait_for(lambda: app.state.queue.stats().waiting == 1)
                    validate_batch(
                        await follower, live_model, inline, workload.references
                    )
                    settled = asdict(app.state.queue.stats())
                    if any(
                        settled[key] for key in ("in_flight", "waiting", "rw_readers")
                    ):
                        raise AssertionError("Mixed input ownership did not settle")
                    if len(children) != 4 or any(
                        child.poll() is None or not child.stdin.closed
                        for child in children
                    ):
                        raise AssertionError(
                            "Remote children were not completed and reaped"
                        )
                    if native_errors or len(completed_batches) != 4:
                        raise AssertionError(
                            "A mixed native computation failed after caller cancellation"
                        )
                finally:
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
    return {
        "batch_size": size,
        "remote_items_per_batch": 2,
        "remote_body_bytes": body_bytes,
        "successful_pair_seconds": success_seconds,
        "queued": queued,
        "detached_owner": retained,
        "settled": settled,
        "child_count": len(children),
        "children_reaped": True,
        "validated_native_batches": len(completed_batches),
        "unauthenticated_status": unauthorized.status_code,
        "transport": "in-process ASGI with probe-only loopback child transport",
    }
