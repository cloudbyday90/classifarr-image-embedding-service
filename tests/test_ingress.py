# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Ingress ownership spans complete responses and precedes all body receives."""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

import httpx2
import pytest
from fakes import FakeEmbedder, _no_auth_settings
from test_request_input_limits import raw_request

from image_embedder.embedder import ImageEmbedder
from image_embedder.ingress import IngressAdmission
from image_embedder.ingress_middleware import IngressAdmissionMiddleware
from image_embedder.main import create_app


def scope(path="/embed-image", method="POST", version="1.1"):
    return {"type": "http", "path": path, "method": method, "http_version": version}


async def never_receive():
    pytest.fail("overloaded request received a body")


def test_parallel_owners_never_exceed_capacity():
    admission = IngressAdmission(7)
    with ThreadPoolExecutor(max_workers=24) as pool:
        owners = list(pool.map(lambda _: admission.try_acquire(), range(100)))
    assert sum(owners) == admission.stats().active == 7
    assert admission.stats().rejected == 93
    for owned in owners:
        if owned:
            admission.release()
    assert admission.stats().active == 0
    with pytest.raises(RuntimeError, match="without an owner"):
        admission.release()


@pytest.mark.parametrize("maximum", [0, -1, True, 1.5])
def test_invalid_admission_capacity_is_refused(maximum):
    with pytest.raises(ValueError, match="positive integer"):
        IngressAdmission(maximum)


@pytest.mark.anyio
@pytest.mark.parametrize("version", ["1.0", "1.1", "2"])
async def test_overload_never_enters_downstream_or_reads_body(version):
    admission = IngressAdmission(1)
    assert admission.try_acquire()

    async def downstream(*_args):
        pytest.fail("overloaded request entered downstream")

    sent = []

    async def send(message):
        sent.append(message)

    await IngressAdmissionMiddleware(downstream, admission)(
        scope(version=version), never_receive, send
    )
    assert sent[0]["status"] == 503
    headers = dict(sent[0]["headers"])
    assert headers[b"retry-after"] == b"1"
    assert (b"connection" in headers) == (version != "2")
    assert json.loads(sent[1]["body"]) == {"detail": "HTTP ingress capacity exceeded"}
    assert admission.stats().active == admission.stats().rejected == 1
    admission.release()


@pytest.mark.anyio
@pytest.mark.parametrize("path", ["/health", "/ready"])
@pytest.mark.parametrize("method", ["GET", "HEAD"])
async def test_only_read_only_probes_bypass_full_admission(path, method):
    admission = IngressAdmission(1)
    assert admission.try_acquire()
    called = []

    async def downstream(request_scope, *_args):
        called.append(request_scope)

    middleware = IngressAdmissionMiddleware(downstream, admission)
    await middleware(scope(path, method), never_receive, never_receive)
    assert called and admission.stats().active == 1
    sent = []

    async def send(message):
        sent.append(message)

    await middleware(scope(path, "POST"), never_receive, send)
    assert sent[0]["status"] == 503
    admission.release()


@pytest.mark.anyio
async def test_non_http_scope_passes_through_without_admission():
    admission = IngressAdmission(1)
    called = []

    async def downstream(request_scope, *_args):
        called.append(request_scope)

    await IngressAdmissionMiddleware(downstream, admission)(
        {"type": "lifespan"}, never_receive, never_receive
    )
    assert called == [{"type": "lifespan"}]
    assert admission.stats().active == 0


@pytest.mark.anyio
@pytest.mark.parametrize("ending", ["error", "disconnect", "cancel", "send-error"])
async def test_terminal_paths_release_request_owner(ending):
    admission = IngressAdmission(1)
    entered = asyncio.Event()

    async def downstream(_scope, receive, send):
        entered.set()
        if ending == "error":
            raise RuntimeError("failure")
        if ending == "cancel":
            await asyncio.Event().wait()
        elif ending == "disconnect":
            assert (await receive())["type"] == "http.disconnect"
        else:
            await send({"type": "http.response.start", "status": 200, "headers": []})

    async def receive():
        return {"type": "http.disconnect"}

    async def send(_message):
        raise OSError("client disconnected during send")

    task = asyncio.create_task(
        IngressAdmissionMiddleware(downstream, admission)(scope(), receive, send)
    )
    await entered.wait()
    if ending == "cancel":
        assert admission.stats().active == 1
        task.cancel()
    if ending == "disconnect":
        await task
    else:
        with pytest.raises((RuntimeError, OSError, asyncio.CancelledError)):
            await task
    assert admission.stats().active == 0
    assert admission.try_acquire()
    admission.release()


@pytest.mark.anyio
async def test_capacity_is_held_through_response_send():
    admission = IngressAdmission(1)
    sending, finish = asyncio.Event(), asyncio.Event()

    async def downstream(_scope, _receive, send):
        await send({"type": "http.response.body", "body": b"done"})

    async def send(_message):
        sending.set()
        await finish.wait()

    middleware = IngressAdmissionMiddleware(downstream, admission)
    task = asyncio.create_task(middleware(scope(), never_receive, send))
    try:
        await sending.wait()
        assert admission.stats().active == 1
        assert not admission.try_acquire()
    finally:
        finish.set()
        await task
    assert admission.stats().active == 0


@pytest.mark.anyio
async def test_complete_upload_blocks_mixed_paths_but_health_and_other_apps_work():
    settings = _no_auth_settings(max_http_requests=1)
    app = create_app(FakeEmbedder(), settings)
    other_app = create_app(FakeEmbedder(), settings)
    receiving, finish = asyncio.Event(), asyncio.Event()

    async def delayed_body():
        receiving.set()
        await finish.wait()
        yield b'{"image_url":"https://example.com/x"}'

    async with httpx2.AsyncClient(
        transport=httpx2.ASGITransport(app), base_url="http://test"
    ) as client:
        upload = asyncio.create_task(
            client.post(
                "/embed-image",
                content=delayed_body(),
                headers={"content-type": "application/json"},
            )
        )
        try:
            await receiving.wait()
            for path in ["/embed-image", "/embed-batch", "/models"]:
                response = await client.post(path, json={})
                assert response.status_code == 503
            for path in ["/health", "/ready"]:
                assert (await client.get(path)).status_code == 200
            assert other_app.state.ingress.try_acquire()
            other_app.state.ingress.release()
        finally:
            finish.set()
            assert (await upload).status_code == 200
        assert (await client.post("/embed-image", json={})).status_code == 422
    assert app.state.ingress.stats().active == 0


@pytest.mark.anyio
async def test_rejected_real_app_body_is_never_received():
    app = create_app(FakeEmbedder(), _no_auth_settings(max_http_requests=1))
    assert app.state.ingress.try_acquire()
    status, receives = await raw_request(app, [b"x" * 1000])
    assert status == 503 and receives == 0
    assert app.state.queue.stats().in_flight == app.state.queue.stats().waiting == 0
    app.state.ingress.release()


@pytest.mark.anyio
@pytest.mark.parametrize(
    "body,status",
    [
        (b"broken", 422),
        (b"{}", 422),
        (b"x" * 33, 413),
        (b'{"image_base64":"YQ=="}', 400),
    ],
)
async def test_admitted_validation_failures_release_capacity(body, status):
    app = create_app(
        ImageEmbedder(_no_auth_settings()),
        _no_auth_settings(max_http_requests=1, max_request_body_bytes=32),
    )
    actual, _ = await raw_request(app, [body])
    assert actual == status and app.state.ingress.stats().active == 0
