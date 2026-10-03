# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Hold complete ingress capacity before EOF, reject excess, then recover."""

import asyncio
from dataclasses import asdict

import httpx
import numpy as np
from capacity_receive_observer import ReceiveObserver
from capacity_socket_server import header_only_status, wait_until
from capacity_upload_body import UploadBody


async def staged_uploads(
    client: httpx.AsyncClient,
    port,
    app,
    observer: ReceiveObserver,
    body: UploadBody,
    model: str,
    expected: list,
    label: str,
    reader,
    emit,
) -> dict:
    count = app.state.settings.max_http_requests
    releases = [asyncio.Event() for _ in range(count)]
    labels = [f"{label}-{index}" for index in range(count)]
    tasks = [
        asyncio.create_task(
            client.post(
                "/embed-batch",
                content=body.stream(release),
                headers={"Content-Length": str(body.length), "X-Capacity-Id": name},
            )
        )
        for release, name in zip(releases, labels)
    ]
    try:
        await wait_until(
            lambda: all(
                observer.records.get(name) is not None
                and observer.records[name].bytes_received == body.length - 1
                for name in labels
            )
        )
        held = asdict(app.state.ingress.stats())
        queue = asdict(app.state.queue.stats())
        if held["active"] != count or queue["in_flight"] or queue["waiting"]:
            raise AssertionError("Held upload dispatched inference or lost ingress")
        emit(
            {
                "event": "socket_upload_held",
                "case": label,
                "ingress": held,
                "received_bytes": [
                    observer.records[name].bytes_received for name in labels
                ],
                "memory": reader(),
            }
        )
        rejected_label = f"{label}-excess"
        status = await header_only_status(
            port, body.length, app.state.settings.service_api_key, rejected_label
        )
        await wait_until(lambda: observer.records[rejected_label].completed)
        if status != 503 or observer.records[rejected_label].bytes_received:
            raise AssertionError("Excess ingress was not rejected before body receive")
        for task in tasks[1:]:
            task.cancel()
        outcomes = await asyncio.gather(*tasks[1:], return_exceptions=True)
        if any(not isinstance(value, asyncio.CancelledError) for value in outcomes):
            raise AssertionError("Held upload completed instead of disconnecting")
        await wait_until(lambda: app.state.ingress.stats().active == 1)
        if any(not observer.records[name].disconnected for name in labels[1:]):
            raise AssertionError("Client cancellation did not reach the real server")
        releases[0].set()
        response = await tasks[0]
        if response.status_code != 200:
            raise AssertionError(
                f"Retained socket upload failed: HTTP {response.status_code}"
            )
        result = response.json()
        if (
            result["total"] != len(expected)
            or result["succeeded"] != len(expected)
            or result["failed"]
            or len(result["results"]) != len(expected)
        ):
            raise AssertionError("Retained upload lost ordered results")
        for index, (row, vector) in enumerate(zip(result["results"], expected)):
            if (
                row["index"] != index
                or row["model"] != model
                or row["dims"] != len(vector)
            ):
                raise AssertionError("Retained upload metadata changed")
            np.testing.assert_allclose(row["embedding"], vector, rtol=1e-4, atol=1e-4)
        await wait_until(lambda: app.state.ingress.stats().active == 0)
        return {
            "case": label,
            "held": held,
            "body_bytes_per_client": body.length,
            "json_bytes": len(body.encoded_json),
            "excess_status": status,
            "disconnected_callers": len(outcomes),
            "retained_status": response.status_code,
            "retained_received_bytes": observer.records[labels[0]].bytes_received,
            "settled": asdict(app.state.queue.stats()),
        }
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await wait_until(
            lambda: app.state.ingress.stats().active == 0,
            timeout=app.state.settings.shutdown_timeout_seconds,
        )
