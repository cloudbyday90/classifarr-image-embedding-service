# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Native parser evidence for the shipped launcher's direct and trusted modes."""

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
    "protocol", [ResponseDeadlineHTTPProtocol, ResponseDeadlineH11Protocol]
)
@pytest.mark.parametrize("peers", [[], ["127.0.0.1"], ["192.0.2.254"]])
@pytest.mark.parametrize("path", ["/health", "/ready"])
async def test_explicit_settings_control_quota_address_and_scheme(
    protocol, peers, path, tmp_path, monkeypatch
):
    monkeypatch.setenv("FORWARDED_ALLOW_IPS", "*")
    app = protected_app(rate_limit_health="2/minute", server_forwarded_allow_ips=peers)
    observations, statuses = [], []

    async def observed(scope, receive, send):
        observations.append((scope["client"][0], scope["scheme"]))
        await app(scope, receive, send)

    settings = app.state.settings
    async with running_server(
        observed,
        app.state.ingress,
        protocol,
        tmp_path,
        forwarded_allow_ips=settings.server_forwarded_allow_ips,
        proxy_headers=bool(settings.server_forwarded_allow_ips),
    ) as (_, port, _, _, client_ssl):
        for address in ["198.51.100.1", "198.51.100.1", "198.51.100.1", "198.51.100.2"]:
            reader, writer = await connect(port, client_ssl)
            try:
                async with asyncio.timeout(5):
                    writer.write(
                        (
                            f"GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n"
                            f"X-Forwarded-For: {address}\r\n"
                            "X-Forwarded-For: 127.0.0.1\r\nX-Forwarded-Proto: https\r\n\r\n"
                        ).encode()
                    )
                    await writer.drain()
                    response = await reader.read()
                statuses.append(int(response.split(b" ", 2)[1]))
            finally:
                await close_writer(writer)
    trusted = peers == ["127.0.0.1"]
    assert statuses == [200, 200, 429, 200 if trusted else 429]
    assert observations == (
        [("198.51.100.1", "https")] * 3 + [("198.51.100.2", "https")]
        if trusted
        else [("127.0.0.1", "http")] * 4
    )
    assert app.state.ingress.stats().active == app.state.embedder.calls == 0
