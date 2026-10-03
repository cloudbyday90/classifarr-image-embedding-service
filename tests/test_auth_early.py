# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Header-only authentication, routing policy and response ownership contracts."""

import asyncio
import base64
import json

import pytest
from fakes import FakeEmbedder, _png_bytes
from fastapi import FastAPI

from image_embedder.config import Settings
from image_embedder.main import create_app


class CountingEmbedder(FakeEmbedder):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def embed(self, *args):
        self.calls += 1
        return super().embed(*args)

    def embed_batch(self, *args):
        self.calls += 1
        return super().embed_batch(*args)


def protected_app(**changes):
    settings = Settings(
        **(
            dict(
                require_api_key=True,
                service_api_key="fixture-key",
                warmup_on_startup=False,
                cleanup_on_shutdown=False,
                max_http_requests=2,
            )
            | changes
        )
    )
    return create_app(CountingEmbedder(), settings)


def request_scope(
    path="/embed-image", method="POST", *, headers=(), root="", version="1.1", query=b""
):
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": version,
        "method": method,
        "path": root + path,
        "raw_path": (root + path).encode(),
        "root_path": root,
        "query_string": query,
        "scheme": "http",
        "client": ("127.0.0.1", 1234),
        "server": ("test", 80),
        "headers": [(b"content-type", b"application/json"), *headers],
    }


async def forbidden_receive():
    pytest.fail("Rejected authentication attempted to receive the request body")


async def raw_response(app, scope, body=None):
    messages, received, finished = [], 0, asyncio.Event()

    async def receive():
        nonlocal received
        if body is None:
            await forbidden_receive()
        received += 1
        if received == 1:
            return {"type": "http.request", "body": body, "more_body": False}
        await finished.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        messages.append(message)
        if message["type"] == "http.response.body" and not message.get(
            "more_body", False
        ):
            finished.set()

    await asyncio.wait_for(app(scope, receive, send), 5)
    start = next(row for row in messages if row["type"] == "http.response.start")
    data = b"".join(row.get("body", b"") for row in messages)
    headers = dict(start["headers"])
    content = (
        json.loads(data)
        if headers.get(b"content-type", b"").startswith(b"application/json")
        else data
    )
    return start["status"], headers, content, received


@pytest.mark.anyio
@pytest.mark.parametrize(
    "path,method",
    [
        ("/embed-image", "POST"),
        ("/embed-batch", "POST"),
        ("/admin/cleanup", "POST"),
        ("/models", "GET"),
    ],
)
@pytest.mark.parametrize(
    "headers", [(), ((b"x-api-key", b"wrong"),), ((b"authorization", b"Bearer wrong"),)]
)
async def test_every_protected_route_rejects_without_receive(path, method, headers):
    app = protected_app()
    status, response_headers, data, received = await raw_response(
        app, request_scope(path, method, headers=headers)
    )
    assert status == 401 and data == {"detail": "Invalid or missing API key"}
    assert response_headers[b"connection"] == b"close" and received == 0
    assert app.state.embedder.calls == app.state.ingress.stats().active == 0
    assert app.state.queue.stats().in_flight == app.state.queue.stats().waiting == 0


@pytest.mark.anyio
@pytest.mark.parametrize("root", ["", "/prefix"])
@pytest.mark.parametrize("configured,status", [("fixture-key", 401), (None, 503)])
async def test_admin_is_mandatory_in_dev_mode_and_under_root_path(
    root, configured, status
):
    app = protected_app(require_api_key=False, service_api_key=configured)
    result = await raw_response(app, request_scope("/admin/cleanup", root=root))
    assert result[0] == status and result[3] == 0


@pytest.mark.anyio
async def test_misconfigured_service_fails_closed_before_body():
    app = protected_app(service_api_key=None)
    status, _, data, received = await raw_response(app, request_scope())
    assert status == 503 and received == 0
    assert data == {"detail": "Service API key is not configured. Set SERVICE_API_KEY."}


@pytest.mark.anyio
@pytest.mark.parametrize(
    "headers",
    [
        ((b"x-api-key", b"wrong"), (b"authorization", b"Bearer fixture-key")),
        ((b"x-api-key", b"\xe9"),),
        ((b"authorization", b"Bearer \xe9"),),
    ],
)
async def test_wrong_primary_and_non_ascii_credentials_remain_401(headers):
    status, _, _, received = await raw_response(
        protected_app(), request_scope(headers=headers)
    )
    assert status == 401 and received == 0


@pytest.mark.anyio
async def test_query_credentials_are_not_accepted():
    result = await raw_response(
        protected_app(), request_scope(query=b"api_key=fixture-key")
    )
    assert result[0] == 401 and result[3] == 0


@pytest.mark.anyio
@pytest.mark.parametrize("version", ["1.0", "1.1", "2"])
async def test_close_header_only_for_http1(version):
    result = await raw_response(protected_app(), request_scope(version=version))
    assert result[0] == 401
    assert (b"connection" in result[1]) == (version != "2")


@pytest.mark.anyio
@pytest.mark.parametrize(
    "headers",
    [
        ((b"x-api-key", b"fixture-key"),),
        ((b"authorization", b"bEaReR fixture-key"),),
        ((b"x-api-key", b""), (b"authorization", b"Bearer fixture-key")),
    ],
)
async def test_accepted_headers_preserve_body_validation_and_success(headers):
    app = protected_app()
    result = await raw_response(app, request_scope(headers=headers), b"{")
    assert result[0] == 422 and result[3] > 0 and app.state.embedder.calls == 0
    body = json.dumps(
        {"image_base64": base64.b64encode(_png_bytes()).decode()}
    ).encode()
    result = await raw_response(app, request_scope(headers=headers), body)
    assert result[0] == 200 and result[2]["dims"] == 768
    assert app.state.embedder.calls == 1 and app.state.ingress.stats().active == 0


@pytest.mark.anyio
async def test_dev_mode_bypasses_auth_on_non_admin_routes():
    result = await raw_response(
        protected_app(require_api_key=False, service_api_key=None),
        request_scope(),
        b"{",
    )
    assert result[0] == 422 and result[3] > 0


@pytest.mark.anyio
async def test_mounted_admin_route_keeps_mandatory_protection():
    inner = protected_app(require_api_key=False)
    outer = FastAPI()
    outer.mount("/service", inner)
    result = await raw_response(outer, request_scope("/service/admin/cleanup"))
    assert result[0] == 401 and result[3] == 0
    assert inner.state.ingress.stats().active == 0


@pytest.mark.anyio
async def test_authentication_policy_is_isolated_between_apps():
    first = protected_app()
    second = protected_app(service_api_key="other-fixture-key")
    scope = request_scope(
        "/models", "GET", headers=((b"x-api-key", b"other-fixture-key"),)
    )
    assert (await raw_response(first, scope))[0] == 401
    assert (await raw_response(second, scope))[0] == 200


@pytest.mark.anyio
@pytest.mark.parametrize(
    "path,method,status",
    [
        ("/unknown", "POST", 404),
        ("/embed-image", "GET", 405),
        ("/embed-image/", "POST", 307),
    ],
)
async def test_unmatched_methods_and_redirects_keep_routing_contract(
    path, method, status
):
    # RedirectResponse has no JSON body, so capture ASGI messages directly.
    app, sent = protected_app(), []

    async def send(message):
        sent.append(message)

    await asyncio.wait_for(app(request_scope(path, method), forbidden_receive, send), 5)
    assert sent[0]["status"] == status and app.state.ingress.stats().active == 0


@pytest.mark.anyio
async def test_ingress_and_declared_body_limit_still_precede_auth():
    app = protected_app(max_request_body_bytes=10, max_http_requests=1)
    result = await raw_response(
        app, request_scope(headers=((b"content-length", b"11"),))
    )
    assert result[0] == 413 and result[3] == 0
    assert app.state.ingress.try_acquire()
    try:
        result = await raw_response(app, request_scope())
        assert result[0] == 503 and result[2] == {
            "detail": "HTTP ingress capacity exceeded"
        }
        public = await raw_response(app, request_scope("/health", "GET"))
        assert public[0] == 200
    finally:
        app.state.ingress.release()


@pytest.mark.anyio
async def test_rejected_response_retains_ingress_until_completion():
    app = protected_app(max_http_requests=1)
    sending, release = asyncio.Event(), asyncio.Event()

    async def blocked_send(message):
        if message["type"] == "http.response.body" and not message.get(
            "more_body", False
        ):
            sending.set()
            await release.wait()

    task = asyncio.create_task(app(request_scope(), forbidden_receive, blocked_send))
    try:
        await asyncio.wait_for(sending.wait(), 5)
        assert app.state.ingress.stats().active == 1
        result = await raw_response(app, request_scope())
        assert result[0] == 503 and result[3] == 0
        release.set()
        await asyncio.wait_for(task, 5)
        assert app.state.ingress.stats().active == 0
        result = await raw_response(
            app,
            request_scope("/models", "GET", headers=((b"x-api-key", b"fixture-key"),)),
            b"",
        )
        assert result[0] == 200
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
