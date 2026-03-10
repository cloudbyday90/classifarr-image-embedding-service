# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""API key authentication and rate-limiting helpers."""

import hmac

from fastapi import Depends, HTTPException, Request
from fastapi.security import APIKeyHeader, APIKeyQuery
from slowapi import Limiter
from slowapi.util import get_remote_address

from .config import Settings

_api_key_header = APIKeyHeader(name="X-Api-Key", auto_error=False)
_bearer_header = APIKeyHeader(name="Authorization", auto_error=False)


def _extract_api_key(
    x_api_key: str | None,
    authorization: str | None,
) -> str | None:
    """Return the API key from X-Api-Key or Authorization: Bearer <key>."""
    if x_api_key:
        return x_api_key
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:]
    return None


def make_auth_dependency(settings: Settings):
    """
    Return a FastAPI dependency that enforces API key authentication.

    - When REQUIRE_API_KEY=true (default): all callers must supply a valid key.
    - When REQUIRE_API_KEY=false (local dev): unauthenticated requests pass through.
    - /admin/cleanup is ALWAYS protected regardless of REQUIRE_API_KEY.
    """

    async def verify_api_key(
        request: Request,
        x_api_key: str | None = Depends(_api_key_header),
        authorization: str | None = Depends(_bearer_header),
    ) -> None:
        path = request.url.path
        is_admin = path.startswith("/admin/")

        # Admin endpoints are always protected.
        if not settings.require_api_key and not is_admin:
            return

        if not settings.service_api_key:
            # Key enforcement is on but SERVICE_API_KEY was not configured — fail
            # closed to avoid a misconfiguration silently opening the service.
            raise HTTPException(
                status_code=503,
                detail="Service API key is not configured. Set SERVICE_API_KEY.",
            )

        candidate = _extract_api_key(x_api_key, authorization)
        if not candidate or not hmac.compare_digest(candidate, settings.service_api_key):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    return verify_api_key


def make_limiter(settings: Settings) -> Limiter:
    """
    Return a slowapi Limiter keyed by API key (falls back to client IP).
    This ensures each caller has an independent quota.
    """

    def _key_func(request: Request) -> str:
        x_api_key = request.headers.get("x-api-key")
        if x_api_key:
            return x_api_key
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            return auth[7:]
        return get_remote_address(request)

    return Limiter(key_func=_key_func)
