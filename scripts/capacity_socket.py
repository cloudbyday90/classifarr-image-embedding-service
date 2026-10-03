# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Authenticated direct HTTP/1 upload capacity using the existing native workload."""

import asyncio
import secrets
from dataclasses import asdict, replace

import httpx
from capacity_receive_observer import ReceiveObserver
from capacity_socket_mixed import mixed_socket_batches
from capacity_socket_server import header_only_status, socket_server, wait_until
from capacity_socket_upload import staged_uploads
from capacity_upload_body import UploadBody, ceiling_image, validate_socket_limits


async def check_socket_capacity(workload) -> dict:
    from image_embedder.main import create_app

    settings = workload.embedder.settings
    validate_socket_limits(settings)
    if not settings.allow_remote_urls or settings.allowed_remote_hosts != [
        "capacity.example"
    ]:
        raise ValueError("Socket experiment requires the fixed probe remote authority")
    fixture_key = secrets.token_urlsafe(24)
    protected = replace(settings, require_api_key=True, service_api_key=fixture_key)
    largest = max(
        workload.models, key=lambda name: workload.embedder.resolve_model(name).dims
    )
    payload, edge = ceiling_image(settings)
    reference = await asyncio.to_thread(
        workload.embedder.embed, None, payload, largest, False, 224
    )
    single = UploadBody.create(
        {"model": largest, "items": [{"image_base64": payload}], "normalize": False},
        settings.max_request_body_bytes,
    )
    size = settings.embed_batch_api_max_items
    batch = UploadBody.create(
        {
            "model": largest,
            "items": [{"image_base64": workload.payloads[i % 2]} for i in range(size)],
            "normalize": False,
        },
        settings.max_request_body_bytes,
    )
    app = create_app(workload.embedder, protected)
    observer = ReceiveObserver(app)
    async with asyncio.timeout(300), socket_server(observer, protected) as port:
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{port}",
            trust_env=False,
            timeout=120,
            headers={
                "X-Api-Key": fixture_key,
                "Content-Type": "application/json",
            },
            limits=httpx.Limits(max_connections=settings.max_http_requests + 2),
        ) as client:
            unauthorized = await header_only_status(
                port, single.length, None, "unauthenticated"
            )
            await wait_until(lambda: observer.records["unauthenticated"].completed)
            if (
                unauthorized != 401
                or observer.records["unauthenticated"].bytes_received
            ):
                raise AssertionError(
                    "Socket authentication did not reject before body receive"
                )
            status = await header_only_status(
                port, single.length + 1, protected.service_api_key, "overflow"
            )
            await wait_until(lambda: observer.records["overflow"].completed)
            if status != 413 or observer.records["overflow"].bytes_received:
                raise AssertionError(
                    "Declared overflow consumed body or changed status"
                )
            staged = []
            for body, vectors, label in (
                (single, [reference[0]], "single-ceilings"),
                (
                    batch,
                    [workload.references[largest][i % 2] for i in range(size)],
                    "batch-ceiling",
                ),
            ):
                staged.append(
                    await staged_uploads(
                        client,
                        port,
                        app,
                        observer,
                        body,
                        largest,
                        vectors,
                        label,
                        workload.reader,
                        workload.emit,
                    )
                )
            mixed = await mixed_socket_batches(workload, app, client, observer)
    ingress, queue = asdict(app.state.ingress.stats()), asdict(app.state.queue.stats())
    if (
        ingress["active"]
        or queue["in_flight"]
        or queue["waiting"]
        or queue["rw_readers"]
    ):
        raise AssertionError("Socket experiment retained ingress/native owners")
    return {
        "transport": "direct-loopback-http1",
        "http_client": "httpx",
        "single_image_source_pixels": edge**2,
        "single_image_bytes": settings.max_image_bytes,
        "maximum_body_bytes": settings.max_request_body_bytes,
        "limits": {
            "http_requests": settings.max_http_requests,
            "server_connections": settings.server_concurrency,
            "image_bytes": settings.max_image_bytes,
            "image_pixels": settings.max_image_pixels,
            "batch_bytes": settings.max_batch_image_bytes,
            "batch_pixels": settings.max_batch_image_pixels,
            "body_bytes": settings.max_request_body_bytes,
            "upload_seconds": settings.request_body_timeout_seconds,
            "response_seconds": settings.response_send_timeout_seconds,
            "embedding_seconds": settings.embedding_timeout_seconds,
            "queue_wait_seconds": settings.embed_max_wait_seconds,
            "remote_batch_seconds": settings.remote_batch_fetch_timeout_seconds,
        },
        "staged": staged,
        "mixed": mixed,
        "settled_ingress": ingress,
        "settled_queue": queue,
        "unauthenticated_status": 401,
        "unauthenticated_received_bytes": observer.records[
            "unauthenticated"
        ].bytes_received,
        "overflow_status": 413,
        "fixture_scope": "client/server/sampler in same process; existing remote child substitution",
    }
