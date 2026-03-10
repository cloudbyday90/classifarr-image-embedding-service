# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for API key authentication and rate limiting."""

import pytest
from fastapi.testclient import TestClient

from image_embedder.main import create_app
from image_embedder.config import Settings


# ── Helpers ──────────────────────────────────────────────────────────────────

def _app(require_api_key: bool = True, api_key: str | None = "secret-key"):
    settings = Settings()
    settings.require_api_key = require_api_key
    settings.service_api_key = api_key
    settings.warmup_on_startup = False
    return create_app(settings=settings)


# ── /health and /ready are always public ─────────────────────────────────────

def test_health_no_key_required():
    client = TestClient(_app())
    assert client.get("/health").status_code == 200


def test_ready_no_key_required():
    client = TestClient(_app())
    assert client.get("/ready").status_code == 200


# ── Endpoints require auth when REQUIRE_API_KEY=true ─────────────────────────

@pytest.mark.parametrize("path,method", [
    ("/models", "get"),
    ("/admin/cleanup", "post"),
])
def test_protected_endpoints_reject_unauthenticated(path, method):
    client = TestClient(_app(), raise_server_exceptions=False)
    resp = getattr(client, method)(path)
    assert resp.status_code == 401


def test_protected_endpoints_accept_x_api_key():
    client = TestClient(_app())
    assert client.get("/models", headers={"X-Api-Key": "secret-key"}).status_code == 200


def test_protected_endpoints_accept_bearer_token():
    client = TestClient(_app())
    assert client.get("/models", headers={"Authorization": "Bearer secret-key"}).status_code == 200


def test_protected_endpoints_reject_wrong_key():
    client = TestClient(_app(), raise_server_exceptions=False)
    assert client.get("/models", headers={"X-Api-Key": "wrong-key"}).status_code == 401


# ── /admin/cleanup is ALWAYS protected, even when REQUIRE_API_KEY=false ──────

def test_admin_always_protected_without_key():
    client = TestClient(_app(require_api_key=False), raise_server_exceptions=False)
    assert client.post("/admin/cleanup").status_code == 401


def test_admin_accessible_with_key_even_when_not_required():
    client = TestClient(_app(require_api_key=False))
    assert client.post("/admin/cleanup", headers={"X-Api-Key": "secret-key"}).status_code == 200


# ── REQUIRE_API_KEY=false allows public access to non-admin endpoints ─────────

def test_dev_mode_allows_unauthenticated_models():
    client = TestClient(_app(require_api_key=False))
    assert client.get("/models").status_code == 200


# ── Misconfigured: REQUIRE_API_KEY=true but SERVICE_API_KEY not set ───────────

def test_missing_service_api_key_returns_503():
    client = TestClient(_app(require_api_key=True, api_key=None), raise_server_exceptions=False)
    resp = client.get("/models", headers={"X-Api-Key": "any-key"})
    assert resp.status_code == 503
