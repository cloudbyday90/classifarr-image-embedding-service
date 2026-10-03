# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Shared container entrypoint with explicit worker and connection budgets."""

import asyncio
from functools import partial
from typing import cast

import uvicorn

from .config import Settings
from .response_http_protocol import ResponseDeadlineHTTPProtocol


def main() -> None:
    settings = Settings()
    uvicorn.run(
        "image_embedder.main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.server_workers,
        limit_concurrency=settings.server_concurrency,
        backlog=settings.server_backlog,
        proxy_headers=bool(settings.server_forwarded_allow_ips),
        forwarded_allow_ips=settings.server_forwarded_allow_ips,
        # uvloop's opaque TLS transport hides queued ciphertext from final drain.
        loop="asyncio",
        # Config accepts a callable protocol factory; partial remains spawn-pickleable.
        http=cast(
            type[asyncio.Protocol],
            partial(
                ResponseDeadlineHTTPProtocol,
                timeout_seconds=settings.response_send_timeout_seconds,
            ),
        ),
    )


if __name__ == "__main__":
    main()
