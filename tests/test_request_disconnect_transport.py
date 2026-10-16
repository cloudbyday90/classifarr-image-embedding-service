# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Real post-upload closes exercise matched routes, admission and coalescing."""

import asyncio
import json

import pytest
from fakes import FakeEmbedder, _no_auth_settings
from test_response_http_protocol import (
    close_writer,
    connect,
    running_server,
    wait_until,
)
from worker_fakes import ThreadGate

from image_embedder.execution_cancellation import cancellation_checkpoint
from image_embedder.main import create_app
from image_embedder.response_http_protocol import (
    ResponseDeadlineH11Protocol,
    ResponseDeadlineHTTPProtocol,
)


@pytest.fixture(params=[False, True], ids=["asyncio", "uvloop"])
def anyio_backend(request):
    if request.param:
        pytest.importorskip("uvloop")
    return "asyncio", {"use_uvloop": request.param}


class CooperativeEmbedder(FakeEmbedder):
    def __init__(self, gate):
        super().__init__()
        self.gate = gate

    def checked(self, function, *args):
        cancellation_checkpoint()
        return function(*args)

    def embed(self, *args):
        return self.gate.run(self.checked, super().embed, *args)

    def embed_batch(self, *args):
        return self.gate.run(self.checked, super().embed_batch, *args)


def make_app(gate, *, coalesced=False):
    events = []

    class Observer:
        def begin(self):
            return self

        def mark(self, phase):
            events.append(phase)

    app = create_app(
        embedder=CooperativeEmbedder(gate),
        settings=_no_auth_settings(
            embedding_timeout_seconds=10,
            rate_limit_embed="1000/minute",
            embed_max_queue=2,
            embed_batch_window_ms=1000 if coalesced else 0,
            embed_batch_max_size=2,
            cleanup_on_shutdown=False,
        ),
        execution_observer=Observer(),
    )
    return app, events


async def post(writer, path="/embed-image", *, partial=False):
    item = {"image_url": "https://example.com/controlled.png"}
    body = json.dumps({"items": [item]} if path == "/embed-batch" else item).encode()
    writer.write(
        f"POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n".encode()
        + (body[:1] if partial else body)
    )
    await writer.drain()
    return body[1:]


async def result(reader):
    header = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 4)
    assert header.startswith(b"HTTP/1.1 200"), header
    headers = dict(line.split(b": ", 1) for line in header.split(b"\r\n")[1:-2])
    return json.loads(await asyncio.wait_for(reader.readexactly(int(headers[b"content-length"])), 4))


@pytest.mark.anyio
@pytest.mark.parametrize("protocol", [ResponseDeadlineHTTPProtocol, ResponseDeadlineH11Protocol])
@pytest.mark.parametrize(
    "tls,anyio_backend", [(False, False), (True, False), (False, True)],
    indirect=["anyio_backend"],
)
@pytest.mark.parametrize("path", ["/embed-image", "/embed-batch"])
async def test_post_upload_disconnect_retains_active_owner_and_recovers(protocol, tls, path, tmp_path, caplog, anyio_backend):
    gate = ThreadGate()
    app, events = make_app(gate)
    try:
        async with app.router.lifespan_context(app):
            async with running_server(app, app.state.ingress, protocol, tmp_path, tls) as (_, port, _, _, client_ssl):
                reader, writer = await connect(port, client_ssl)
                try:
                    await post(writer, path)
                    await gate.wait_started()
                    await close_writer(writer)
                    await wait_until(lambda: "waiter_detached" in events and app.state.ingress.stats().active == 0)
                    assert gate.active == app.state.queue.stats().in_flight == app.state.queue.stats().rw_readers == 1
                    recovery_reader, recovery_writer = await connect(port, client_ssl)
                    try:
                        await post(recovery_writer, path)
                        await wait_until(lambda: app.state.queue.stats().waiting == 1)
                        gate.release.set()
                        response = await result(recovery_reader)
                        row = response["results"][0] if path == "/embed-batch" else response
                        assert row["dims"] == 768 and row["embedding"] == [0.1] * 768
                        await wait_until(lambda: app.state.ingress.stats().active == 0)
                        assert "native_cancelled" in events
                        assert app.state.queue.stats().in_flight == app.state.queue.stats().rw_readers == 0
                    finally:
                        await close_writer(recovery_writer)
                finally:
                    gate.release.set()
                    await close_writer(writer)
    finally:
        gate.release.set()
        assert await app.state.executor.close(2)
    assert "ERROR" not in caplog.text


@pytest.mark.anyio
@pytest.mark.parametrize("protocol", [ResponseDeadlineHTTPProtocol, ResponseDeadlineH11Protocol])
@pytest.mark.parametrize("path", ["/embed-image", "/embed-batch"])
async def test_post_upload_queued_disconnect_removes_admission(protocol, path, tmp_path):
    gate = ThreadGate()
    app, events = make_app(gate)
    blocker = asyncio.create_task(app.state.executor.run(gate.run, lambda: 42))
    try:
        await gate.wait_started()
        async with running_server(app, app.state.ingress, protocol, tmp_path) as (_, port, _, _, _):
            _, writer = await connect(port)
            try:
                await post(writer, path)
                await wait_until(lambda: app.state.queue.stats().waiting == 1)
                await close_writer(writer)
                await wait_until(lambda: app.state.queue.stats().waiting == app.state.ingress.stats().active == 0)
                assert gate.calls == 1 and app.state.queue.stats().in_flight == 1
                gate.release.set()
                assert await blocker == 42
                assert events.count("dispatched") == 1
            finally:
                gate.release.set()
                await close_writer(writer)
    finally:
        gate.release.set()
        await blocker
        assert await app.state.executor.close(2)


@pytest.mark.anyio
@pytest.mark.parametrize("protocol", [ResponseDeadlineHTTPProtocol, ResponseDeadlineH11Protocol])
@pytest.mark.parametrize("last_leaves", [False, True])
async def test_coalesced_disconnect_preserves_live_member_and_stops_last(protocol, last_leaves, tmp_path):
    gate = ThreadGate()
    app, events = make_app(gate, coalesced=True)
    try:
        async with app.router.lifespan_context(app):
            async with running_server(app, app.state.ingress, protocol, tmp_path) as (_, port, _, _, _):
                _, first = await connect(port)
                reader, second = await connect(port)
                try:
                    await post(first)
                    await post(second)
                    await gate.wait_started()
                    await close_writer(first)
                    await wait_until(lambda: app.state.ingress.stats().active == 1)
                    assert "waiter_detached" not in events and gate.calls == 1
                    if last_leaves:
                        await close_writer(second)
                        await wait_until(lambda: "waiter_detached" in events and app.state.ingress.stats().active == 0)
                    assert gate.active == app.state.queue.stats().in_flight == app.state.queue.stats().rw_readers == 1
                    gate.release.set()
                    if not last_leaves:
                        assert (await result(reader))["embedding"] == [0.1] * 768
                    await wait_until(lambda: app.state.queue.stats().in_flight == app.state.ingress.stats().active == 0)
                    assert ("native_cancelled" in events) == last_leaves
                finally:
                    gate.release.set()
                    await close_writer(first)
                    await close_writer(second)
    finally:
        gate.release.set()
        assert await app.state.executor.close(2)


@pytest.mark.anyio
@pytest.mark.parametrize("protocol", [ResponseDeadlineHTTPProtocol, ResponseDeadlineH11Protocol])
async def test_slow_body_then_successful_keepalive_never_has_competing_receive(protocol, tmp_path):
    gate = ThreadGate()
    app, _ = make_app(gate)
    active, peak = 0, 0

    async def observed(scope, receive, send):
        async def single_receive():
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            try:
                return await receive()
            finally:
                active -= 1
        await app(scope, single_receive, send)

    gate.release.set()
    async with running_server(observed, app.state.ingress, protocol, tmp_path) as (_, port, _, _, _):
        reader, writer = await connect(port)
        try:
            rest = await post(writer, partial=True)
            await wait_until(lambda: active == 1)
            assert gate.calls == 0
            writer.write(rest)
            await writer.drain()
            assert (await result(reader))["dims"] == 768
            await post(writer)
            assert (await result(reader))["dims"] == 768
            await wait_until(lambda: app.state.ingress.stats().active == 0)
            assert peak == 1 and active == 0 and gate.calls == 2
        finally:
            await close_writer(writer)
    assert await app.state.executor.close(2)
