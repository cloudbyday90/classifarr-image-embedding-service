# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Concurrent mixed/inline socket batches with real supervised remote children."""

import asyncio
import base64
from dataclasses import asdict

from capacity_mixed import validate_batch
from capacity_remote_fixture import remote_fixture
from capacity_socket_server import wait_until
from capacity_upload_body import UploadBody


async def mixed_socket_batches(workload, app, client, observer) -> dict:
    settings = app.state.settings
    size = settings.embed_batch_api_max_items
    largest = max(
        workload.models, key=lambda name: workload.embedder.resolve_model(name).dims
    )
    live = next((name for name in workload.models if name != largest), largest)
    inline = [{"image_base64": workload.payloads[i % 2]} for i in range(size)]
    inline_bytes = sum(
        len(base64.b64decode(row["image_base64"], validate=True)) for row in inline[2:]
    )
    remote_bytes = min(
        settings.max_image_bytes, (settings.max_batch_image_bytes - inline_bytes) // 2
    )
    mixed = [
        {"image_url": f"http://capacity.example/{i}"} if i < 2 else inline[i]
        for i in range(size)
    ]
    tasks = []
    with remote_fixture(workload.payloads, remote_bytes) as (started, children):
        try:

            async def submit(model, items, normalize, label):
                body = UploadBody.create(
                    {"model": model, "items": items, "normalize": normalize},
                    settings.max_request_body_bytes,
                )
                return await client.post(
                    "/embed-batch",
                    content=body.stream(),
                    headers={
                        "Content-Length": str(body.length),
                        "X-Capacity-Id": label,
                    },
                )

            first = asyncio.create_task(submit(largest, mixed, False, "mixed"))
            tasks.append(first)
            await wait_until(started.is_set)
            second = asyncio.create_task(submit(live, inline, True, "inline"))
            tasks.append(second)
            await wait_until(lambda: app.state.queue.stats().waiting == 1)
            queued = asdict(app.state.queue.stats())
            responses = await asyncio.gather(first, second)
            validate_batch(responses[0], largest, mixed, workload.references, False)
            validate_batch(responses[1], live, inline, workload.references, True)
            await wait_until(lambda: app.state.ingress.stats().active == 0)
            settled = asdict(app.state.queue.stats())
            if settled["in_flight"] or settled["waiting"] or settled["rw_readers"]:
                raise AssertionError("Socket batches retained native ownership")
            if len(children) != 2 or any(child.poll() is None for child in children):
                raise AssertionError("Socket remote children were not reaped")
            return {
                "statuses": [response.status_code for response in responses],
                "queued": queued,
                "settled": settled,
                "batch_size": size,
                "body_bytes_per_client": settings.max_request_body_bytes,
                "received_bytes": {
                    name: observer.records[name].bytes_received
                    for name in ("mixed", "inline")
                },
                "remote_bytes_per_image": remote_bytes,
                "children_reaped": True,
                "child_count": len(children),
            }
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
