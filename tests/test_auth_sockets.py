# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Actual HTTP/1 parsers/TLS reject header-only uploads and recover capacity."""

import asyncio
import base64
import json

import pytest
from capacity_receive_observer import ReceiveObserver
from fakes import _png_bytes
from test_auth_early import protected_app
from test_response_http_protocol import (
    close_writer,
    connect,
    running_server,
    wait_until,
)

from image_embedder.response_http_protocol import (
    ResponseDeadlineH11Protocol,
    ResponseDeadlineHTTPProtocol,
)


@pytest.mark.anyio
@pytest.mark.parametrize(
    "protocol",
    [ResponseDeadlineHTTPProtocol, ResponseDeadlineH11Protocol],
    ids=["httptools", "h11"],
)
@pytest.mark.parametrize("tls", [False, True], ids=["plain", "tls"])
@pytest.mark.parametrize("path", ["/embed-image", "/embed-batch"])
@pytest.mark.parametrize(
    "configured,credential,status",
    [
        ("fixture-key", None, 401),
        ("fixture-key", "wrong", 401),
        (None, "fixture-key", 503),
    ],
)
async def test_header_only_auth_rejection_has_no_continue_or_body_and_closes(
    protocol, tls, path, configured, credential, status, tmp_path
):
    app = protected_app(service_api_key=configured)
    observer = ReceiveObserver(app)
    async with running_server(
        observer, app.state.ingress, protocol, tmp_path, tls=tls
    ) as (_, port, _, _, client_ssl):
        reader, writer = await connect(port, client_ssl)
        try:
            authorization = f"X-Api-Key: {credential}\r\n" if credential else ""
            request = (
                f"POST {path} HTTP/1.1\r\nHost: localhost\r\n"
                f"Content-Length: {app.state.settings.max_request_body_bytes}\r\n"
                "Content-Type: application/json\r\nExpect: 100-continue\r\n"
                "X-Capacity-Id: rejected\r\nConnection: keep-alive\r\n"
                + authorization
                + "\r\n"
            )
            async with asyncio.timeout(5):
                writer.write(request.encode())
                await writer.drain()
                response = (
                    await reader.read()
                )  # EOF proves server-side connection closure.
            header, body = response.split(b"\r\n\r\n", 1)
            assert header.startswith(f"HTTP/1.1 {status} ".encode())
            assert (
                b"100 Continue" not in response
                and b"connection: close" in header.lower()
            )
            expected = (
                "Invalid or missing API key"
                if status == 401
                else "Service API key is not configured. Set SERVICE_API_KEY."
            )
            assert json.loads(body) == {"detail": expected}
            await wait_until(lambda: observer.records["rejected"].completed)
            assert (
                observer.records["rejected"].bytes_received
                == app.state.embedder.calls
                == 0
            )
            await wait_until(lambda: app.state.ingress.stats().active == 0)
        finally:
            await close_writer(writer)
        if configured is not None:
            reader, writer = await connect(port, client_ssl)
            try:
                item = {"image_base64": base64.b64encode(_png_bytes()).decode()}
                payload = json.dumps(
                    {"items": [item]} if path == "/embed-batch" else item
                ).encode()
                async with asyncio.timeout(5):
                    writer.write(
                        (
                            f"POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Length: {len(payload)}\r\n"
                            "Content-Type: application/json\r\nAuthorization: Bearer fixture-key\r\n"
                            "X-Capacity-Id: recovered\r\nConnection: close\r\n\r\n"
                        ).encode()
                        + payload
                    )
                    await writer.drain()
                    response = await reader.read()
                header, body = response.split(b"\r\n\r\n", 1)
                assert header.startswith(b"HTTP/1.1 200 ")
                data = json.loads(body)
                row = data["results"][0] if path == "/embed-batch" else data
                assert row["dims"] == len(row["embedding"]) == 768
                await wait_until(lambda: app.state.ingress.stats().active == 0)
                assert observer.records["recovered"].bytes_received == len(payload)
                assert app.state.embedder.calls == 1
                assert (
                    app.state.queue.stats().in_flight
                    == app.state.queue.stats().waiting
                    == 0
                )
            finally:
                await close_writer(writer)
