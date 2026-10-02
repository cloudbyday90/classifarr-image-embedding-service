# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Actual HTTP/TLS transport through a recorded numeric connection seam."""

import gzip
import socket
import ssl
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import trustme

from image_embedder.config import Settings
from image_embedder.input_limits import InputLimitExceeded
from image_embedder.remote_fetch import RemoteFetchError, fetch_remote_image


@contextmanager
def local_server(
    monkeypatch, tmp_path, *, tls=False, trusted=True, hostname="images.example"
):
    seen, sni = [], []
    body = b"abc"

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append((self.path, dict(self.headers)))
            data = gzip.compress(b"x" * 1000) if self.path == "/gzip" else body
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            if self.path == "/gzip":
                self.send_header("Content-Encoding", "gzip")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    if tls:
        ca = trustme.CA()
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ca.issue_cert(hostname).configure_cert(context)
        context.set_servername_callback(lambda _sock, name, _ctx: sni.append(name))
        server.socket = context.wrap_socket(server.socket, server_side=True)
        if trusted:
            bundle = tmp_path / "test-ca.pem"
            ca.cert_pem.write_to_path(bundle)
            monkeypatch.setattr(
                "image_embedder.remote_fetch.certifi.where", lambda: str(bundle)
            )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    original_dns = socket.getaddrinfo
    original_connect = socket.socket.connect
    lookups, connections = [], []
    port = server.server_port

    def resolve(host, requested_port, *args, **kwargs):
        lookups.append(host)
        if host == "images.example":
            # A second hostname resolution would rebind to loopback.
            address = "8.8.8.8" if lookups.count(host) == 1 else "127.0.0.1"
            return [
                (
                    socket.AF_INET,
                    socket.SOCK_STREAM,
                    socket.IPPROTO_TCP,
                    "",
                    (address, requested_port),
                )
            ]
        assert host == "8.8.8.8", f"Unapproved destination: {host}"
        return original_dns(host, requested_port, *args, **kwargs)

    def connect(sock, address):
        connections.append(address)
        assert address == ("8.8.8.8", port), f"Unapproved socket target: {address}"
        # Test-only mapping to our owned server; production receives no such mapping.
        return original_connect(sock, ("127.0.0.1", port))

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    monkeypatch.setattr(socket.socket, "connect", connect)
    try:
        yield port, seen, sni, lookups, connections
    finally:
        server.shutdown()
        thread.join(3)
        server.server_close()
        assert not thread.is_alive()


@pytest.mark.parametrize("tls", [False, True])
def test_real_numeric_transport_preserves_host_tls_and_ignores_ambient_auth(
    monkeypatch, tmp_path, tls
):
    netrc = tmp_path / ".netrc"
    netrc.write_text(
        "machine images.example login secret-user password secret-password\n"
    )
    monkeypatch.setenv("NETRC", str(netrc))
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("ALL_PROXY", "socks5://127.0.0.1:9")
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", "/nonexistent/environment-bundle")
    with local_server(monkeypatch, tmp_path, tls=tls) as (
        port,
        seen,
        sni,
        lookups,
        connections,
    ):
        scheme = "https" if tls else "http"
        result = fetch_remote_image(
            f"{scheme}://images.example:{port}/poster?token=a%2Fb",
            Settings(allow_remote_urls=True, max_image_bytes=3),
        )
        assert result == b"abc"
        assert lookups == ["images.example", "8.8.8.8"]
        assert connections == [("8.8.8.8", port)]
        assert seen[0][0] == "/poster?token=a%2Fb"
        assert seen[0][1]["Host"] == f"images.example:{port}"
        assert (
            "Authorization" not in seen[0][1]
            and "Proxy-Authorization" not in seen[0][1]
        )
        assert "Cookie" not in seen[0][1]
        assert sni == (["images.example"] if tls else [])


@pytest.mark.parametrize(
    "trusted,hostname", [(False, "images.example"), (True, "wrong.example")]
)
def test_real_tls_rejects_untrusted_ca_and_wrong_hostname(
    monkeypatch, tmp_path, trusted, hostname
):
    with local_server(
        monkeypatch, tmp_path, tls=True, trusted=trusted, hostname=hostname
    ) as (port, seen, sni, lookups, connections):
        with pytest.raises(RemoteFetchError, match="Unable to fetch") as error:
            fetch_remote_image(
                f"https://images.example:{port}/poster?secret=token",
                Settings(allow_remote_urls=True),
            )
        assert "secret" not in str(error.value)
        assert not seen and sni == ["images.example"]
        assert lookups == ["images.example", "8.8.8.8"]
        assert connections == [("8.8.8.8", port)]


def test_real_gzip_body_is_limited_by_decoded_bytes(monkeypatch, tmp_path):
    with local_server(monkeypatch, tmp_path) as (port, *_rest):
        with pytest.raises(InputLimitExceeded):
            fetch_remote_image(
                f"http://images.example:{port}/gzip",
                Settings(allow_remote_urls=True, max_image_bytes=80),
            )
