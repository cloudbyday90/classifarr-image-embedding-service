# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""API-key authentication policy."""

from fastapi import Depends, HTTPException, Request
from fastapi.security import APIKeyHeader

from .config import Settings
from .credentials import extract_api_key, matches_api_key

_api_key_header = APIKeyHeader(name="X-Api-Key", auto_error=False)
_bearer_header = APIKeyHeader(name="Authorization", auto_error=False)


def make_auth_dependency(settings: Settings, *, always_required: bool = False):
    """
    Return a FastAPI dependency that enforces API key authentication.

    - When REQUIRE_API_KEY=true (default): all callers must supply a valid key.
    - When REQUIRE_API_KEY=false (local dev): unauthenticated requests pass through.
    - /admin/cleanup is ALWAYS protected regardless of REQUIRE_API_KEY.
    - always_required binds mandatory protection to a router, including mounts.
    """

    async def verify_api_key(
        request: Request,
        x_api_key: str | None = Depends(_api_key_header),
        authorization: str | None = Depends(_bearer_header),
    ) -> None:
        path = request.url.path
        is_admin = always_required or path.startswith("/admin/")

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

        candidate = extract_api_key(x_api_key, authorization)
        if not matches_api_key(candidate, settings.service_api_key):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    return verify_api_key
