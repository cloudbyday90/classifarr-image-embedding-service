# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Offline native sockets prove held bodies, rejection, disconnect and recovery."""

import asyncio
import base64
import json
from io import BytesIO

import pytest
from capacity_socket import check_socket_capacity
from capacity_socket_server import socket_server
from capacity_upload_body import UploadBody, ceiling_image, validate_socket_limits
from capacity_workload import make_payloads
from PIL import Image
from test_capacity_workload import FixtureEmbedder, workload

from image_embedder import remote_process
from image_embedder.config import Settings


def socket_settings(**changes):
    values = dict(
        allow_remote_urls=True,
        allowed_remote_hosts=["capacity.example"],
        max_image_bytes=8192,
        max_image_pixels=1024,
        max_request_body_bytes=32768,
        max_http_requests=2,
        embed_batch_api_max_items=4,
        warmup_on_startup=False,
        cleanup_on_shutdown=False,
    )
    return Settings(**(values | changes))


def test_ceiling_fixture_is_valid_at_exact_byte_and_square_pixel_limits():
    settings = socket_settings()
    payload, edge = ceiling_image(settings)
    data = base64.b64decode(payload, validate=True)
    assert (
        len(data) == settings.max_image_bytes and edge**2 == settings.max_image_pixels
    )
    with Image.open(BytesIO(data)) as image:
        image.load()
        assert image.mode == "RGB" and image.size == (edge, edge)


def test_upload_body_is_repeatable_exact_and_valid_json():
    request = {"items": [{"image_base64": "fixture"}]}
    body = UploadBody.create(request, 1024)
    first = b"".join(body.chunks(body.length, 31))
    assert len(first) == 1024 and json.loads(first) == request
    assert first == b"".join(body.chunks(body.length, 17))


def test_upload_stream_holds_exactly_one_byte_before_eof():
    async def run():
        body = UploadBody.create({"value": 1}, 128)
        release = asyncio.Event()
        stream = body.stream(release)
        first = await anext(stream)
        pending = asyncio.create_task(anext(stream))
        # The JSON prefix and padding are separate bounded chunks.
        padding = await pending
        assert len(first + padding) == body.length - 1
        last = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        assert not last.done()
        release.set()
        assert len(first + padding + await last) == body.length
        with pytest.raises(StopAsyncIteration):
            await anext(stream)

    asyncio.run(run())


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_http_requests", 17),
        ("max_image_pixels", 32_000_001),
        ("max_image_bytes", 16 * 1024**2 + 1),
        ("server_concurrency", 4),
    ],
)
def test_probe_refuses_unbounded_or_transport_masked_pressure(field, value):
    with pytest.raises(ValueError, match="bounded"):
        validate_socket_limits(socket_settings(**{field: value}))


def test_incompatible_fixture_budgets_fail_before_server_start():
    with pytest.raises(ValueError, match="image byte ceiling"):
        ceiling_image(socket_settings(max_image_bytes=1))
    with pytest.raises(ValueError, match="request body ceiling"):
        UploadBody.create({"items": ["too large"]}, 1)


def test_server_closes_when_caller_fails():
    async def app(scope, receive, send):
        if scope["type"] == "lifespan":
            while True:
                message = await receive()
                await send({"type": message["type"] + ".complete"})
                if message["type"] == "lifespan.shutdown":
                    return

    async def run():
        with pytest.raises(ValueError, match="caller failure"):
            async with socket_server(app, socket_settings()) as port:
                assert port > 0
                raise ValueError("caller failure")
        with pytest.raises(OSError):
            await asyncio.open_connection("127.0.0.1", port)

    asyncio.run(run())


def test_native_socket_capacity_with_real_children_and_ambient_proxy(monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://invalid-proxy.test:9")
    monkeypatch.setenv("SSL_CERT_FILE", "/missing/socket-fixture-certificate.pem")
    events = []
    subject = workload(events)
    settings = socket_settings()

    class RemoteEmbedder(FixtureEmbedder):
        def embed_batch(self, spec, size, items):
            for item in items:
                if item.image_url:
                    data = remote_process.fetch_remote_image(item.image_url, settings)
                    assert len(data) == settings.max_image_bytes
            return super().embed_batch(spec, size, items)

    subject.embedder = RemoteEmbedder()
    subject.embedder.settings = settings
    subject.payloads = make_payloads(32)
    subject.load(False)
    command = remote_process._command
    result = asyncio.run(check_socket_capacity(subject))
    assert result["transport"] == "direct-loopback-http1"
    assert result["unauthenticated_status"] == 401 and result["overflow_status"] == 413
    assert result["unauthenticated_received_bytes"] == 0
    for stage in result["staged"]:
        assert stage["held"]["active"] == 2 and stage["excess_status"] == 503
        assert stage["disconnected_callers"] == 1 and stage["retained_status"] == 200
        assert stage["retained_received_bytes"] == settings.max_request_body_bytes
        assert stage["settled"]["in_flight"] == stage["settled"]["waiting"] == 0
    assert result["mixed"]["child_count"] == 2 and result["mixed"]["children_reaped"]
    assert result["mixed"]["queued"]["waiting"] == 1
    assert result["settled_ingress"]["active"] == 0
    assert remote_process._command is command
    assert sum(record["event"] == "socket_upload_held" for record in events) == 2
