# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Actual Uvicorn forwarding trust remains the quota-address boundary."""

import asyncio

import pytest
from test_auth_early import protected_app
from test_response_http_protocol import close_writer, connect, running_server

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
@pytest.mark.parametrize(
    "trust", ["192.0.2.254", "127.0.0.1"], ids=["untrusted-peer", "trusted-peer"]
)
@pytest.mark.parametrize("path", ["/health", "/ready"])
async def test_native_probe_quota_uses_server_trusted_address(
    protocol, trust, path, tmp_path
):
    app = protected_app(rate_limit_health="2/minute")
    statuses = []
    async with running_server(
        app, app.state.ingress, protocol, tmp_path, forwarded_allow_ips=trust
    ) as (_, port, _, _, client_ssl):
        for i, forwarded in enumerate(
            ["198.51.100.1", "198.51.100.1", "198.51.100.1", "198.51.100.2"]
        ):
            reader, writer = await connect(port, client_ssl)
            try:
                async with asyncio.timeout(5):
                    writer.write(
                        (
                            f"GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n"
                            f"X-Api-Key: untrusted-{i}\r\n"
                            f"X-Forwarded-For: {forwarded}\r\n"
                            "X-Forwarded-For: 127.0.0.1\r\n\r\n"
                        ).encode()
                    )
                    await writer.drain()
                    response = await reader.read()
                statuses.append(int(response.split(b" ", 2)[1]))
            finally:
                await close_writer(writer)
    assert statuses == [200, 200, 429, 200 if trust == "127.0.0.1" else 429]
    assert len(app.state.limiter._storage.storage) == (2 if trust == "127.0.0.1" else 1)
    assert app.state.embedder.calls == app.state.ingress.stats().active == 0
