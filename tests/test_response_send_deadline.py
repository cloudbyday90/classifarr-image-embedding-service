# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Response budgets preserve closure/owner ordering and the original error contract."""

import asyncio
import pickle

import anyio
import pytest

from image_embedder.ingress import IngressAdmission
from image_embedder.ingress_middleware import IngressAdmissionMiddleware
from image_embedder.response_send_deadline import ResponseSendDeadline


def scope():
    return {
        "type": "http",
        "method": "POST",
        "path": "/embed-image",
        "http_version": "1.1",
    }


@pytest.fixture
def anyio_backend():
    return "asyncio"


async def unused():
    pytest.fail("unexpected receive")


@pytest.mark.anyio
@pytest.mark.parametrize("mode", ["send", "drain", "gap", "trickle"])
async def test_total_budget_closes_before_cancellation_and_owner_release(mode):
    admission = IngressAdmission(1)
    closed, cleaned, sent = [], [], []

    async def abort():
        assert admission.stats().active == 1 or closed
        await anyio.sleep(0.01)
        closed.append(True)

    async def drain():
        if mode == "drain":
            await anyio.sleep_forever()

    async def send(message):
        sent.append(message)
        if mode == "send":
            await anyio.sleep_forever()

    async def downstream(_scope, _receive, send):
        try:
            await send({"type": "http.response.start", "status": 200, "headers": []})
            if mode == "gap":
                await anyio.sleep_forever()
            if mode == "trickle":
                while True:
                    await send(
                        {"type": "http.response.body", "body": b"x", "more_body": True}
                    )
                    await anyio.sleep(0.01)
            await send({"type": "http.response.body", "body": b"ok"})
        finally:
            with anyio.CancelScope(shield=True):
                assert closed and admission.stats().active == 1
                await anyio.sleep(0.01)
                cleaned.append(True)

    app = ResponseSendDeadline(
        IngressAdmissionMiddleware(downstream, admission), 0.07, abort, drain
    )
    with anyio.fail_after(1):
        await app(scope(), unused, send)
    assert cleaned and admission.stats().active == 0
    assert [m.get("status") for m in sent if m["type"] == "http.response.start"] == [
        200
    ]
    assert len(sent) > 2 if mode == "trickle" else len(sent) <= 2


@pytest.mark.anyio
async def test_budget_starts_at_headers_and_stops_after_terminal_drain():
    sent, drains = [], []

    async def abort():
        pytest.fail("successful response must not abort")

    async def drain():
        drains.append(True)

    async def downstream(_scope, _receive, send):
        await anyio.sleep(0.04)
        await send({"type": "http.response.start", "status": 201, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})
        await anyio.sleep(
            0.04
        )  # completed-response background work has its own lifetime

    async def send(message):
        sent.append(message)

    await ResponseSendDeadline(downstream, 0.02, abort, drain)(scope(), unused, send)
    assert drains == [True] and sent[-1]["body"] == b"ok"


@pytest.mark.anyio
async def test_trailers_remain_within_budget_until_terminal_trailer():
    closed = []

    async def operation():
        closed.append(True)

    async def send(_message):
        pass

    async def downstream(_scope, _receive, send):
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [],
                "trailers": True,
            }
        )
        await send({"type": "http.response.body", "body": b"ok"})
        await send(
            {"type": "http.response.trailers", "headers": [], "more_trailers": True}
        )
        await anyio.sleep_forever()

    with anyio.fail_after(1):
        await ResponseSendDeadline(downstream, 0.02, operation, operation)(
            scope(), unused, send
        )
    assert closed


@pytest.mark.anyio
async def test_expired_cleanup_cannot_send_a_second_response():
    sent = []

    async def operation():
        await anyio.sleep(0)

    async def send(message):
        sent.append(message)

    async def downstream(_scope, _receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        try:
            await anyio.sleep_forever()
        except anyio.get_cancelled_exc_class():
            with anyio.CancelScope(shield=True):
                await send(
                    {"type": "http.response.start", "status": 500, "headers": []}
                )

    await ResponseSendDeadline(downstream, 0.02, operation, operation)(
        scope(), unused, send
    )
    assert len(sent) == 1


@pytest.mark.anyio
@pytest.mark.parametrize("at_send", [False, True])
async def test_original_application_and_send_errors_are_preserved(at_send):
    closed = []

    async def operation():
        closed.append(True)

    async def send(_message):
        raise RuntimeError("original send error")

    async def downstream(_scope, _receive, send):
        if at_send:
            await send({"type": "http.response.start", "status": 200, "headers": []})
        raise ValueError("original application error")

    with pytest.raises(RuntimeError if at_send else ValueError, match="original"):
        await ResponseSendDeadline(downstream, 1, operation, operation)(
            scope(), unused, send
        )
    assert bool(closed) == at_send


@pytest.mark.anyio
async def test_external_cancellation_aborts_a_blocked_send_before_releasing_owner():
    admission = IngressAdmission(1)
    entered = asyncio.Event()
    closed = []

    async def operation():
        closed.append(True)
        assert admission.stats().active == 1 or len(closed) > 1

    async def send(_message):
        entered.set()
        await anyio.sleep_forever()

    async def downstream(_scope, _receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})

    app = ResponseSendDeadline(
        IngressAdmissionMiddleware(downstream, admission), 10, operation, operation
    )
    task = asyncio.create_task(app(scope(), unused, send))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed and admission.stats().active == 0


@pytest.mark.anyio
async def test_non_http_scope_passes_through():
    calls = []

    async def downstream(request_scope, *_args):
        calls.append(request_scope)

    await ResponseSendDeadline(downstream, 1, unused, unused)(
        {"type": "lifespan"}, unused, unused
    )
    assert calls == [{"type": "lifespan"}]


@pytest.mark.anyio
async def test_overdue_clock_is_checked_even_before_the_watcher_can_run(monkeypatch):
    now, sent, closed = [10.0], [], []
    monkeypatch.setattr(anyio, "current_time", lambda: now[0])

    async def operation():
        closed.append(True)

    async def send(message):
        sent.append(message)

    async def downstream(_scope, _receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        now[0] += 2
        await send({"type": "http.response.body", "body": b"overdue"})

    await ResponseSendDeadline(downstream, 1, operation, operation)(
        scope(), unused, send
    )
    assert closed and len(sent) == 1


def test_launcher_keeps_explicit_budgets_and_spawn_pickleable_protocol_factory(
    monkeypatch,
):
    from image_embedder import server

    monkeypatch.setenv("RESPONSE_SEND_TIMEOUT_SECONDS", "0.75")
    calls = []
    monkeypatch.setattr(
        server.uvicorn, "run", lambda *args, **kwargs: calls.append((args, kwargs))
    )
    server.main()
    args, kwargs = calls[0]
    assert args == ("image_embedder.main:app",)
    assert kwargs["loop"] == "asyncio"
    assert (
        kwargs["workers"] == 1
        and kwargs["limit_concurrency"] == 64
        and kwargs["backlog"] == 128
    )
    factory = pickle.loads(pickle.dumps(kwargs["http"]))
    assert factory.func is server.ResponseDeadlineHTTPProtocol
    assert factory.keywords == {"timeout_seconds": 0.75}
