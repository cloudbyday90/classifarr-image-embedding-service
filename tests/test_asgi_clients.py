# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Client migration contracts for state, exceptions and lifespan ownership."""

from contextlib import asynccontextmanager

import httpx2
import pytest
from asgi_clients import lifespan_client
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient


@pytest.fixture
def anyio_backend():
    return "asyncio"


def stateful_app(events: list[str]) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        events.append("startup")
        try:
            yield {"owner": "lifespan"}
        finally:
            events.append("shutdown")

    app = FastAPI(lifespan=lifespan)

    @app.post("/state")
    async def state(request: Request):
        return {
            "owner": getattr(request.state, "owner", None),
            "body": await request.json(),
            "header": request.headers.get("x-test-contract"),
        }

    @app.get("/error")
    async def error():
        raise RuntimeError("client contract failure")

    return app


def test_starlette_selects_httpx2_and_drives_lifespan():
    events: list[str] = []
    with TestClient(stateful_app(events)) as client:
        assert isinstance(client, httpx2.Client)
        response = client.post("/state", json={"value": 1})
        assert isinstance(response, httpx2.Response)
        assert response.status_code == 200
        assert response.json()["owner"] == "lifespan"
        assert events == ["startup"]
    assert events == ["startup", "shutdown"]


@pytest.mark.anyio
async def test_managed_client_carries_state_and_ignores_ambient_tls(monkeypatch):
    monkeypatch.setenv("SSL_CERT_FILE", "/missing/client-contract-certificate.pem")
    monkeypatch.setenv("HTTPS_PROXY", "http://invalid-proxy.test:9")
    events: list[str] = []
    async with lifespan_client(
        stateful_app(events), headers={"X-Test-Contract": "present"}
    ) as client:
        response = await client.post("/state", json={"value": 2})
        assert isinstance(response, httpx2.Response)
        assert response.status_code == 200
        assert response.json() == {
            "owner": "lifespan",
            "body": {"value": 2},
            "header": "present",
        }
        assert events == ["startup"]
    assert client.is_closed
    assert events == ["startup", "shutdown"]


@pytest.mark.anyio
async def test_managed_client_closes_when_caller_raises():
    events: list[str] = []
    with pytest.raises(ValueError, match="caller failure"):
        async with lifespan_client(stateful_app(events)) as client:
            assert (await client.post("/state", json={})).status_code == 200
            raise ValueError("caller failure")
    assert client.is_closed
    assert events == ["startup", "shutdown"]


@pytest.mark.anyio
@pytest.mark.parametrize("raise_app_exceptions", [True, False])
async def test_managed_client_preserves_app_exception_policy(raise_app_exceptions):
    events: list[str] = []
    async with lifespan_client(
        stateful_app(events), raise_app_exceptions=raise_app_exceptions
    ) as client:
        if raise_app_exceptions:
            with pytest.raises(RuntimeError, match="client contract failure"):
                await client.get("/error")
        else:
            response = await client.get("/error")
            assert response.status_code == 500
            assert response.text == "Internal Server Error"
    assert client.is_closed
    assert events == ["startup", "shutdown"]


@pytest.mark.anyio
async def test_raw_asgi_transport_does_not_claim_lifespan():
    events: list[str] = []
    transport = httpx2.ASGITransport(app=stateful_app(events))
    async with httpx2.AsyncClient(
        transport=transport, base_url="http://test", trust_env=False
    ) as client:
        response = await client.post("/state", json={})
        assert response.json()["owner"] is None
    assert events == []
