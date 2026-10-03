# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""State-aware API test clients with explicit lifespan and client closure."""

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager

import httpx2
from asgi_lifespan import LifespanManager
from starlette.types import ASGIApp


@asynccontextmanager
async def lifespan_client(
    app: ASGIApp,
    *,
    raise_app_exceptions: bool = True,
    headers: Mapping[str, str] | None = None,
) -> AsyncIterator[httpx2.AsyncClient]:
    async with LifespanManager(app) as manager:
        transport = httpx2.ASGITransport(
            app=manager.app, raise_app_exceptions=raise_app_exceptions
        )
        async with httpx2.AsyncClient(
            transport=transport,
            base_url="http://test",
            headers=headers,
            trust_env=False,
        ) as client:
            yield client
