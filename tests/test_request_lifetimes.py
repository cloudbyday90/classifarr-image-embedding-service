# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Upload expiry cancels cooperative work without dropping complete owners."""

import asyncio
import json

import anyio
import pytest
from fakes import FakeEmbedder, _no_auth_settings
from test_ingress import scope

from image_embedder import config
from image_embedder.ingress import IngressAdmission
from image_embedder.ingress_middleware import IngressAdmissionMiddleware
from image_embedder.main import create_app
from image_embedder.request_body_deadline import RequestBodyDeadlineMiddleware


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
@pytest.mark.parametrize("version", ["1.1", "2"])
@pytest.mark.parametrize("trickle", [False, True])
async def test_upload_deadline_is_total_and_owner_spans_cleanup_and_408(
    version, trickle
):
    admission = IngressAdmission(1)
    sent, receives, cleaned = [], [], []

    async def downstream(_scope, receive, _send):
        try:
            while True:
                await receive()
        finally:
            with anyio.CancelScope(shield=True):
                await anyio.sleep(0.02)
                assert admission.stats().active == 1
                cleaned.append(True)

    async def receive():
        if not trickle:
            await anyio.sleep_forever()
        await anyio.sleep(0.01)
        receives.append(True)
        return {"type": "http.request", "body": b"x", "more_body": True}

    async def send(message):
        assert cleaned and admission.stats().active == 1
        sent.append(message)

    app = IngressAdmissionMiddleware(
        RequestBodyDeadlineMiddleware(downstream, 0.07), admission
    )
    with anyio.fail_after(1):
        await app(scope(version=version), receive, send)
    assert bool(receives) == trickle
    assert sent[0]["status"] == 408
    assert (b"connection" in dict(sent[0]["headers"])) == (version == "1.1")
    assert json.loads(sent[1]["body"]) == {"detail": "Request body timed out"}
    assert admission.stats().active == 0


@pytest.mark.anyio
@pytest.mark.parametrize("completion", ["body", "disconnect", "response"])
async def test_body_or_response_completion_disables_upload_budget(completion):
    sent = []

    async def downstream(_scope, receive, send):
        if completion == "response":
            await send({"type": "http.response.start", "status": 200, "headers": []})
        else:
            await receive()
        await anyio.sleep(0.05)
        if completion != "response":
            await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    async def receive():
        return (
            {"type": "http.disconnect"}
            if completion == "disconnect"
            else {"type": "http.request", "body": b"{}", "more_body": False}
        )

    async def send(message):
        sent.append(message)

    await RequestBodyDeadlineMiddleware(downstream, 0.01)(scope(), receive, send)
    assert [m["status"] for m in sent if m["type"] == "http.response.start"] == [200]


@pytest.mark.anyio
async def test_external_cancellation_and_errors_are_not_translated_to_408():
    entered = asyncio.Event()

    async def downstream(*_args):
        entered.set()
        await anyio.sleep_forever()

    async def unused(*_args):
        pytest.fail("external cancellation must not send 408")

    middleware = RequestBodyDeadlineMiddleware(downstream, 10)
    task = asyncio.create_task(middleware(scope(), unused, unused))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    async def broken(*_args):
        raise RuntimeError("ordinary failure")

    with pytest.raises(RuntimeError, match="ordinary failure"):
        await RequestBodyDeadlineMiddleware(broken, 1)(scope(), unused, unused)


@pytest.mark.anyio
async def test_expired_downstream_cannot_start_a_second_response():
    sent = []

    async def downstream(_scope, _receive, send):
        try:
            await anyio.sleep_forever()
        except anyio.get_cancelled_exc_class():
            with anyio.CancelScope(shield=True):
                await send(
                    {"type": "http.response.start", "status": 200, "headers": []}
                )

    async def unused():
        pytest.fail("unexpected receive")

    async def send(message):
        sent.append(message)

    await RequestBodyDeadlineMiddleware(downstream, 0.01)(scope(), unused, send)
    assert [m["status"] for m in sent if m["type"] == "http.response.start"] == [408]


@pytest.mark.anyio
async def test_non_http_scope_passes_through():
    calls = []

    async def downstream(scope, *_args):
        calls.append(scope)

    await RequestBodyDeadlineMiddleware(downstream, 1)({"type": "lifespan"}, None, None)
    assert calls == [{"type": "lifespan"}]


@pytest.mark.anyio
@pytest.mark.parametrize("path", ["/embed-image", "/embed-batch"])
async def test_real_application_stalled_upload_releases_ingress_without_inference(path):
    app = create_app(
        FakeEmbedder(), _no_auth_settings(request_body_timeout_seconds=0.05)
    )
    request_scope = {
        **scope(path),
        "asgi": {"version": "3.0"},
        "root_path": "",
        "raw_path": path.encode(),
        "query_string": b"",
        "scheme": "http",
        "client": ("127.0.0.1", 1234),
        "server": ("test", 80),
        "headers": [(b"content-type", b"application/json")],
    }
    sent = []

    async def receive():
        await anyio.sleep_forever()

    async def send(message):
        assert app.state.ingress.stats().active == 1
        sent.append(message)

    with anyio.fail_after(2):
        await app(request_scope, receive, send)
    assert sent[0]["status"] == 408
    assert app.state.ingress.stats().active == 0
    assert app.state.queue.stats().in_flight == 0


@pytest.mark.parametrize(
    "name,env,section,default",
    [
        ("request_body_timeout_seconds", "REQUEST_BODY_TIMEOUT_SECONDS", "server", 30),
        ("response_send_timeout_seconds", "RESPONSE_SEND_TIMEOUT_SECONDS", "server", 30),
        ("remote_fetch_timeout_seconds", "REMOTE_FETCH_TIMEOUT_SECONDS", "image", 15),
        ("request_timeout_seconds", "REQUEST_TIMEOUT_SECONDS", "image", 15),
    ],
)
def test_input_budget_defaults_and_configuration_precedence(
    monkeypatch, name, env, section, default
):
    monkeypatch.setattr(config, "_TOML", {})
    monkeypatch.delenv(env, raising=False)
    assert getattr(config.Settings(), name) == default
    monkeypatch.setattr(config, "_TOML", {section: {name: 0.75}})
    assert getattr(config.Settings(), name) == 0.75
    monkeypatch.setenv(env, "0.5")
    assert getattr(config.Settings(), name) == 0.5
    assert getattr(config.Settings(**{name: 0.25}), name) == 0.25


@pytest.mark.parametrize(
    "name,section",
    [
        ("request_body_timeout_seconds", "server"),
        ("response_send_timeout_seconds", "server"),
        ("remote_fetch_timeout_seconds", "image"),
        ("request_timeout_seconds", "image"),
    ],
)
@pytest.mark.parametrize("value", [0, -1, True, float("nan"), float("inf"), "bad"])
def test_invalid_input_budgets_fail_constructor_env_and_toml(
    monkeypatch, name, section, value
):
    monkeypatch.setattr(config, "_TOML", {})
    with pytest.raises(ValueError, match="positive finite"):
        config.Settings(**{name: value})
    monkeypatch.setenv(name.upper(), str(value))
    with pytest.raises(ValueError, match="positive finite"):
        config.Settings()
    monkeypatch.delenv(name.upper())
    monkeypatch.setattr(config, "_TOML", {section: {name: value}})
    with pytest.raises(ValueError, match="positive finite"):
        config.Settings()
