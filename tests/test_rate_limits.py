# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Quota enforcement across unverified headers, public probes and dev mode."""

import base64
import json

import pytest
from fakes import _png_bytes
from fastapi import FastAPI, Request
from test_auth_early import protected_app, raw_response, request_scope

from image_embedder.rate_limits import client_address_key


def scope(path="/health", *, client=("192.0.2.1", 1234), headers=(), root=""):
    result = request_scope(path, "GET", headers=headers, root=root)
    result["client"] = client
    return result


def stored_keys(app):
    return list(app.state.limiter._storage.storage)


@pytest.mark.anyio
@pytest.mark.parametrize("path", ["/health", "/ready"])
@pytest.mark.parametrize("mode", [True, False])
@pytest.mark.parametrize("root", ["", "/prefix"])
async def test_public_probe_headers_cannot_select_another_identity(path, mode, root):
    app = protected_app(require_api_key=mode, rate_limit_health="2/minute")
    variants = [
        (),
        ((b"x-api-key", b"untrusted-a"),),
        ((b"x-api-key", b"untrusted-b"),),
        ((b"authorization", b"Bearer untrusted-c"),),
        ((b"authorization", b"bEaReR untrusted-d"),),
        ((b"authorization", b"Bearer "),),
        ((b"authorization", b"Basic untrusted-e"),),
        ((b"x-api-key", b"fixture-key"),),
        ((b"authorization", b"Bearer fixture-key"),),
        ((b"x-api-key", b""), (b"authorization", b"Bearer fixture-key")),
        ((b"x-api-key", b"\xe9"),),
        ((b"x-api-key", b"untrusted-f"), (b"authorization", b"Bearer fixture-key")),
        ((b"x-forwarded-for", b"198.51.100.99"),),
        ((b"forwarded", b"for=198.51.100.100"),),
        ((b"x-real-ip", b"198.51.100.101"),),
    ]
    statuses = [
        (await raw_response(app, scope(path, headers=headers, root=root)))[0]
        for headers in variants
    ]
    assert statuses == [200, 200] + [429] * (len(variants) - 2)
    assert len(stored_keys(app)) == 1
    assert "fixture-key" not in stored_keys(app)[0]
    assert app.state.embedder.calls == app.state.ingress.stats().active == 0


@pytest.mark.anyio
async def test_probe_addresses_ports_and_endpoint_scopes_remain_independent():
    app = protected_app(rate_limit_health="1/minute")
    assert (await raw_response(app, scope()))[0] == 200
    assert (await raw_response(app, scope(client=("192.0.2.1", 5678))))[0] == 429
    assert (await raw_response(app, scope(client=("192.0.2.2", 5678))))[0] == 200
    assert (await raw_response(app, scope("/ready")))[0] == 200
    assert (await raw_response(app, scope("/ready")))[0] == 429
    assert len(stored_keys(app)) == 3


@pytest.mark.parametrize("client", [None, ("", 12), ("::1", 12)])
def test_address_identity_is_nonempty_without_a_client_or_with_ipv6(client):
    request = Request(scope(client=client))
    assert client_address_key(request)
    assert client_address_key(request) == client_address_key(
        Request(scope(client=client))
    )


@pytest.mark.anyio
async def test_mounted_probe_policy_and_app_state_are_independent():
    first, second = [protected_app(rate_limit_health="1/minute") for _ in range(2)]
    outer = FastAPI()
    outer.mount("/service", first)
    assert (await raw_response(outer, scope("/service/health")))[0] == 200
    assert (await raw_response(outer, scope("/service/health")))[0] == 429
    assert (await raw_response(second, scope()))[0] == 200
    assert len(stored_keys(first)) == len(stored_keys(second)) == 1


@pytest.mark.anyio
@pytest.mark.parametrize("path", ["/embed-image", "/embed-batch"])
@pytest.mark.parametrize("configured", [None, "fixture-key"])
async def test_dev_mode_unverified_and_empty_credentials_share_address_quota(
    path, configured
):
    app = protected_app(
        require_api_key=False, service_api_key=configured, rate_limit_embed="2/minute"
    )
    item = {"image_base64": base64.b64encode(_png_bytes()).decode()}
    body = json.dumps({"items": [item]} if path == "/embed-batch" else item).encode()
    variants = [
        ((b"authorization", b"Bearer "),),
        ((b"x-api-key", b"wrong-a"),),
        ((b"authorization", b"Bearer wrong-b"),),
        ((b"x-api-key", b"\xe9"),),
        ((b"x-api-key", b"wrong-c"), (b"authorization", b"Bearer fixture-key")),
        (),
    ]
    statuses = []
    for headers in variants:
        request = request_scope(path, headers=headers)
        statuses.append((await raw_response(app, request, body))[0])
    assert statuses == [200, 200, 429, 429, 429, 429]
    assert len(stored_keys(app)) == 1 and app.state.embedder.calls == 2
    assert app.state.queue.stats().in_flight == app.state.queue.stats().waiting == 0
    assert app.state.ingress.stats().active == 0


@pytest.mark.anyio
@pytest.mark.parametrize("mode", [True, False])
async def test_verified_headers_share_principal_and_do_not_leak_into_storage(
    mode, caplog
):
    app = protected_app(require_api_key=mode, rate_limit_embed="2/minute")
    body = json.dumps(
        {"image_base64": base64.b64encode(_png_bytes()).decode()}
    ).encode()
    variants = [
        ((b"x-api-key", b"fixture-key"),),
        ((b"authorization", b"bEaReR fixture-key"),),
        ((b"x-api-key", b""), (b"authorization", b"Bearer fixture-key")),
    ]
    results = []
    for i, headers in enumerate(variants):
        request = request_scope(headers=headers)
        request["client"] = (f"192.0.2.{i + 1}", 1234)
        results.append(await raw_response(app, request, body))
    assert [result[0] for result in results] == [200, 200, 429]
    assert all(result[2]["dims"] == 768 for result in results[:2])
    assert len(stored_keys(app)) == 1
    assert "fixture-key" not in str(stored_keys(app)) + caplog.text
    assert app.state.embedder.calls == 2


@pytest.mark.anyio
async def test_probe_exhaustion_does_not_consume_embedding_quota():
    app = protected_app(rate_limit_health="1/minute", rate_limit_embed="1/minute")
    assert (await raw_response(app, scope()))[0] == 200
    assert (await raw_response(app, scope()))[0] == 429
    item = {"image_base64": base64.b64encode(_png_bytes()).decode()}
    for path, payload in [("/embed-image", item), ("/embed-batch", {"items": [item]})]:
        result = await raw_response(
            app,
            request_scope(path, headers=((b"x-api-key", b"fixture-key"),)),
            json.dumps(payload).encode(),
        )
        assert result[0] == 200
    assert len(stored_keys(app)) == 3


@pytest.mark.anyio
async def test_expired_probe_window_recovers_with_same_identity(monkeypatch):
    app = protected_app(rate_limit_health="1/minute")
    assert (await raw_response(app, scope()))[0] == 200
    assert (await raw_response(app, scope()))[0] == 429
    storage = app.state.limiter._storage
    key = stored_keys(app)[0]
    from limits.storage import memory

    deadline = storage.expirations[key]
    monkeypatch.setattr(memory.time, "time", lambda: deadline + 1)
    assert (await raw_response(app, scope(headers=((b"authorization", b"Bearer "),))))[
        0
    ] == 200
    assert stored_keys(app) == [key]
