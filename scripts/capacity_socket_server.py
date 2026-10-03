# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Own a temporary loopback server with the production HTTP protocol/budgets."""

import asyncio
import socket
from contextlib import asynccontextmanager
from functools import partial
from typing import cast

import uvicorn
from starlette.types import ASGIApp

from image_embedder.config import Settings
from image_embedder.response_http_protocol import ResponseDeadlineHTTPProtocol


async def wait_until(predicate, timeout: float = 10) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.005)


@asynccontextmanager
async def socket_server(app: ASGIApp, settings: Settings):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        config = uvicorn.Config(
            app,
            loop="asyncio",
            lifespan="on",
            log_config=None,
            access_log=False,
            http=cast(
                type[asyncio.Protocol],
                partial(
                    ResponseDeadlineHTTPProtocol,
                    timeout_seconds=settings.response_send_timeout_seconds,
                ),
            ),
            limit_concurrency=settings.server_concurrency,
            backlog=settings.server_backlog,
            proxy_headers=bool(settings.server_forwarded_allow_ips),
            forwarded_allow_ips=settings.server_forwarded_allow_ips,
            timeout_graceful_shutdown=settings.shutdown_timeout_seconds,
        )
        server = uvicorn.Server(config)
        task = asyncio.create_task(server.serve(sockets=[listener]))
        try:
            await wait_until(lambda: server.started or task.done())
            if task.done():
                await task
                raise RuntimeError("Capacity server exited before startup")
            yield port
        finally:
            server.should_exit = True
            try:
                async with asyncio.timeout(settings.shutdown_timeout_seconds + 10):
                    await task
            finally:
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
            if server.server_state.connections or server.server_state.tasks:
                raise AssertionError("Capacity server retained HTTP connections/tasks")


async def header_only_status(
    port: int, length: int, key: str | None, label: str
) -> int:
    reader, writer = await asyncio.open_connection("127.0.0.1", port, limit=65536)
    try:
        headers = (
            f"POST /embed-batch HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: {length}\r\n"
            f"Content-Type: application/json\r\nX-Capacity-Id: {label}\r\nConnection: close\r\n"
        )
        if key is not None:
            headers += f"X-Api-Key: {key}\r\n"
        async with asyncio.timeout(10):
            writer.write((headers + "\r\n").encode("ascii"))
            await writer.drain()
            response = await reader.readuntil(b"\r\n\r\n")
        return int(response.split(b"\r\n", 1)[0].split()[1])
    finally:
        writer.close()
        await writer.wait_closed()
