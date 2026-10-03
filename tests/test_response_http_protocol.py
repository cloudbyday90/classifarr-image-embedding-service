# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Real Uvicorn sockets exercise buffered writes, disconnects and capacity recovery."""

import asyncio
import os
import socket
import ssl
from contextlib import asynccontextmanager
from functools import partial

import anyio
import pytest
import trustme
import uvicorn

from image_embedder.ingress import IngressAdmission
from image_embedder.ingress_middleware import IngressAdmissionMiddleware
from image_embedder.response_http_protocol import (
    ResponseDeadlineH11Protocol,
    ResponseDeadlineHTTPProtocol,
)


@pytest.fixture(params=[False, True], ids=["asyncio", "uvloop"])
def anyio_backend(request):
    if request.param:
        pytest.importorskip("uvloop")
    return "asyncio", {"use_uvloop": request.param}


async def wait_until(predicate):
    with anyio.fail_after(4):
        while not predicate():
            await anyio.sleep(0.005)


@asynccontextmanager
async def running_server(app, admission, protocol, tmp_path, tls=False):
    instances, lost = [], []

    class ObservedProtocol(protocol):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.buffer_observations = []

        def _pending_write_bytes(self):
            pending = super()._pending_write_bytes()
            if pending:
                self.buffer_observations.append((pending, admission.stats().active))
            return pending

        def connection_made(self, transport):
            super().connection_made(transport)
            transport.get_extra_info("socket").setsockopt(
                socket.SOL_SOCKET, socket.SO_SNDBUF, 16384
            )
            instances.append(self)

        def connection_lost(self, exc):
            super().connection_lost(exc)
            lost.append((self, admission.stats().active))

    tls_settings = {}
    client_ssl = None
    if tls:
        ca = trustme.CA()
        cert = ca.issue_cert("localhost")
        cert.cert_chain_pems[0].write_to_path(tmp_path / "cert.pem")
        cert.private_key_pem.write_to_path(tmp_path / "key.pem")
        tls_settings = {
            "ssl_certfile": str(tmp_path / "cert.pem"),
            "ssl_keyfile": str(tmp_path / "key.pem"),
        }
        client_ssl = ssl.create_default_context()
        ca.configure_trust(client_ssl)
    config = uvicorn.Config(
        app,
        http=partial(ObservedProtocol, timeout_seconds=0.2),
        lifespan="off",
        log_level="error",
        access_log=False,
        log_config=None,
        timeout_graceful_shutdown=1,
        **tls_settings,
    )
    server = uvicorn.Server(config)
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    task = asyncio.create_task(server.serve(sockets=[listener]))
    try:
        await wait_until(lambda: server.started or task.done())
        assert server.started
        yield server, port, instances, lost, client_ssl
    finally:
        server.should_exit = True
        with anyio.fail_after(4):
            await task
        listener.close()
        assert not server.server_state.connections
        assert not server.server_state.tasks
        assert admission.stats().active == 0


async def connect(port, client_ssl=None):
    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
    sock.setblocking(False)
    await asyncio.get_running_loop().sock_connect(sock, ("127.0.0.1", port))
    return await asyncio.open_connection(
        sock=sock, ssl=client_ssl, server_hostname="localhost" if client_ssl else None
    )


async def request(writer, path="/large", close=False):
    connection = "close" if close else "keep-alive"
    writer.write(
        f"GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: {connection}\r\n\r\n".encode()
    )
    await writer.drain()


async def close_writer(writer):
    if not writer.transport.is_closing():
        writer.transport.abort()
    try:
        with anyio.fail_after(1):
            await writer.wait_closed()
    except (ConnectionError, ssl.SSLError):
        pass


@pytest.mark.anyio
@pytest.mark.parametrize(
    "protocol", [ResponseDeadlineHTTPProtocol, ResponseDeadlineH11Protocol]
)
@pytest.mark.parametrize("tls", [False, True])
@pytest.mark.parametrize("mode", ["terminal", "stream"])
async def test_slow_reader_is_aborted_before_ingress_release_and_capacity_recovers(
    protocol, tls, mode, tmp_path, anyio_backend, caplog
):
    admission = IngressAdmission(1)
    cleaned = []

    async def downstream(scope, _receive, send):
        if scope["path"] == "/quick":
            await send(
                {
                    "type": "http.response.start",
                    "status": 201,
                    "headers": [(b"content-length", b"2")],
                }
            )
            await send({"type": "http.response.body", "body": b"ok"})
            return
        try:
            await send({"type": "http.response.start", "status": 200, "headers": []})
            if mode == "terminal":
                await anyio.sleep(0.01)
                await send(
                    {
                        "type": "http.response.body",
                        "body": b"x" * (8 * 1024 * 1024),
                    }
                )
            else:
                while True:
                    await send(
                        {
                            "type": "http.response.body",
                            "body": b"x" * 65536,
                            "more_body": True,
                        }
                    )
        finally:
            with anyio.CancelScope(shield=True):
                assert admission.stats().active == 1
                cleaned.append(True)

    app = IngressAdmissionMiddleware(downstream, admission)
    async with running_server(app, admission, protocol, tmp_path, tls) as (
        server,
        port,
        instances,
        lost,
        client_ssl,
    ):
        reader, writer = await connect(port, client_ssl)
        try:
            await request(writer)
            opaque_tls = tls and anyio_backend[1]["use_uvloop"]
            if not opaque_tls and not (os.name == "nt" and mode == "terminal"):
                await wait_until(
                    lambda: (
                        instances
                        and (
                            instances[0].buffer_observations
                            or instances[0]._pending_write_bytes() > 0
                        )
                    )
                )
                assert any(
                    active == 1 for _, active in instances[0].buffer_observations
                )
            await wait_until(lambda: cleaned and not server.server_state.tasks)
            if opaque_tls and mode == "terminal":
                assert "observable TLS socket buffer" in caplog.text
            if os.name == "nt" and mode == "terminal" and not lost:
                # Winsock can drain into OS buffers despite an unread client.
                # Peer receipt is separate from the local buffer-drain contract.
                assert instances[0]._pending_write_bytes() == 0
            else:
                assert lost[0] == (instances[0], 1)
                assert instances[0] not in server.server_state.connections
                await instances[0]._abort()  # repeated cleanup after real closure
            assert admission.stats().active == 0
            # No synthetic error response is appended after the original headers.
            partial_body = await reader.read(1024)
            assert partial_body.startswith(b"HTTP/1.1 200")
            assert partial_body.count(b"HTTP/1.1") == 1
        finally:
            await close_writer(writer)
        reader, writer = await connect(port, client_ssl)
        try:
            await request(writer, "/quick", close=True)
            with anyio.fail_after(2):
                data = await reader.read()
            assert data.startswith(b"HTTP/1.1 201") and data.endswith(b"ok")
        finally:
            await close_writer(writer)


@pytest.mark.anyio
@pytest.mark.parametrize(
    "protocol", [ResponseDeadlineHTTPProtocol, ResponseDeadlineH11Protocol]
)
@pytest.mark.parametrize("mode", ["gap", "trickle", "disconnect"])
async def test_stream_gaps_trickle_and_disconnect_finish_owned_work(
    protocol, mode, tmp_path
):
    admission = IngressAdmission(1)
    cleaned = []

    async def downstream(_scope, _receive, send):
        try:
            await send({"type": "http.response.start", "status": 200, "headers": []})
            while True:
                await send(
                    {"type": "http.response.body", "body": b"x", "more_body": True}
                )
                await anyio.sleep(0.01 if mode == "trickle" else 10)
        finally:
            cleaned.append(True)

    async with running_server(
        IngressAdmissionMiddleware(downstream, admission), admission, protocol, tmp_path
    ) as (server, port, instances, lost, _ssl):
        reader, writer = await connect(port)
        try:
            await request(writer)
            with anyio.fail_after(1):
                assert (await reader.read(1024)).startswith(b"HTTP/1.1 200")
            if mode == "disconnect":
                await close_writer(writer)
            await wait_until(lambda: cleaned and not server.server_state.tasks)
            assert instances[0] not in server.server_state.connections
            assert lost and admission.stats().active == 0
        finally:
            await close_writer(writer)


@pytest.mark.anyio
@pytest.mark.parametrize(
    "protocol", [ResponseDeadlineHTTPProtocol, ResponseDeadlineH11Protocol]
)
async def test_successful_terminal_response_bytes_and_keepalive_are_preserved(
    protocol, tmp_path
):
    admission = IngressAdmission(1)
    body = bytes(range(256)) * 256

    async def downstream(_scope, _receive, send):
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    (b"content-length", str(len(body)).encode()),
                    (b"x-test", b"preserved"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    async with running_server(
        IngressAdmissionMiddleware(downstream, admission), admission, protocol, tmp_path
    ) as (server, port, instances, lost, _ssl):
        reader, writer = await connect(port)
        try:
            for _ in range(2):
                await request(writer)
                with anyio.fail_after(2):
                    headers = await reader.readuntil(b"\r\n\r\n")
                    assert (
                        headers.startswith(b"HTTP/1.1 200")
                        and b"x-test: preserved" in headers
                    )
                    assert await reader.readexactly(len(body)) == body
                await wait_until(lambda: not server.server_state.tasks)
                assert admission.stats().active == 0 and not lost
            assert len(instances) == 1
        finally:
            await close_writer(writer)


@pytest.mark.anyio
@pytest.mark.parametrize(
    "protocol", [ResponseDeadlineHTTPProtocol, ResponseDeadlineH11Protocol]
)
async def test_real_starlette_stream_cancels_and_cleans_before_owner_release(
    protocol, tmp_path
):
    from fastapi import Depends, FastAPI
    from starlette.responses import StreamingResponse

    admission = IngressAdmission(1)
    cleaned = []
    app = FastAPI()

    async def owned_resource():
        try:
            yield
        finally:
            with anyio.CancelScope(shield=True):
                assert admission.stats().active == 1
                await anyio.sleep(0.01)
                cleaned.append(True)

    @app.get("/stream", dependencies=[Depends(owned_resource)])
    async def stream():
        async def body():
            while True:
                yield b"x"
                await anyio.sleep(0.01)

        return StreamingResponse(body())

    async with running_server(
        IngressAdmissionMiddleware(app, admission), admission, protocol, tmp_path
    ) as (server, port, instances, lost, _ssl):
        reader, writer = await connect(port)
        try:
            await request(writer, "/stream")
            with anyio.fail_after(1):
                assert (await reader.read(1024)).startswith(b"HTTP/1.1 200")
            await wait_until(lambda: cleaned and not server.server_state.tasks)
            assert lost[0] == (instances[0], 1)
        finally:
            await close_writer(writer)
