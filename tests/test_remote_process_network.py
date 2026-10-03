# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Fresh worker processes against controlled native HTTP/TLS trickle servers."""

import json
import socketserver
import ssl
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import trustme
from test_remote_process import harness

from image_embedder.config import Settings
from image_embedder.input_limits import InputLimitExceeded
from image_embedder.remote_fetch import RemoteFetchError
from image_embedder.remote_process import fetch_remote_image


@contextmanager
def controlled_server(
    tmp_path, *, tls=False, trusted=True, hostname="images.example", stall=False
):
    stopped = threading.Event()
    seen, sni = [], []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append((self.path, dict(self.headers)))
            try:
                if self.path.startswith("/redirect"):
                    if stopped.wait(0.4):
                        return
                    self.send_response(302)
                    self.send_header("Location", self.path + "a")
                    self.end_headers()
                elif self.path == "/headers":
                    for byte in b"HTTP/1.1 200 OK\r\nContent-Length: 4096\r\n\r\n":
                        if stopped.wait(0.04):
                            return
                        self.wfile.write(bytes([byte]))
                        self.wfile.flush()
                else:
                    body = b"abc" if self.path == "/success" else b"x" * 4096
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    if self.path == "/success":
                        self.wfile.write(body)
                    else:
                        for byte in body:
                            if stopped.wait(0.04):
                                return
                            self.wfile.write(bytes([byte]))
                            self.wfile.flush()
            except (OSError, ssl.SSLError):
                pass

        def log_message(self, *_args):
            pass

    class StalledTLS(socketserver.BaseRequestHandler):
        def handle(self):
            self.request.recv(4096)
            seen.append(("TLS ClientHello", {}))
            stopped.wait(5)

    server = (
        socketserver.ThreadingTCPServer(("127.0.0.1", 0), StalledTLS)
        if stall
        else ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    )
    server.daemon_threads = True
    certificate = ""
    if tls and not stall:
        ca = trustme.CA()
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ca.issue_cert(hostname).configure_cert(context)
        context.set_servername_callback(lambda _sock, name, _ctx: sni.append(name))
        server.socket = context.wrap_socket(server.socket, server_side=True)
        if trusted:
            bundle = tmp_path / "ca.pem"
            ca.cert_pem.write_to_path(bundle)
            certificate = str(bundle)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1], certificate, seen, sni
    finally:
        stopped.set()
        server.shutdown()
        server.server_close()
        thread.join(3)
        assert not thread.is_alive()


@pytest.mark.parametrize("phase", ["headers", "body", "redirect", "tls"])
def test_native_trickle_redirect_and_tls_share_total_budget(
    monkeypatch, tmp_path, phase
):
    with controlled_server(tmp_path, stall=phase == "tls") as (
        port,
        certificate,
        seen,
        _sni,
    ):
        children = harness(monkeypatch, tmp_path, "network", port, certificate)
        scheme = "https" if phase == "tls" else "http"
        started = time.monotonic()
        with pytest.raises(RemoteFetchError, match="timed out"):
            fetch_remote_image(
                f"{scheme}://images.example:{port}/{phase}",
                Settings(
                    allow_remote_urls=True,
                    request_timeout_seconds=2,
                    remote_fetch_timeout_seconds=1.2,
                ),
            )
        assert 1.1 <= time.monotonic() - started < 3
        assert seen and children[0].poll() is not None and children[0].stdin.closed
        if phase == "redirect":
            assert len(seen) >= 2


@pytest.mark.parametrize("tls", [False, True])
def test_native_worker_success_preserves_host_sni_and_strips_ambient_state(
    monkeypatch, tmp_path, tls
):
    for name in (
        "SERVICE_API_KEY",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NETRC",
        "PYTHONPATH",
        "SSL_CERT_FILE",
    ):
        monkeypatch.setenv(name, "test-ambient-value")
    with controlled_server(tmp_path, tls=tls) as (port, certificate, seen, sni):
        children = harness(monkeypatch, tmp_path, "network", port, certificate)
        scheme = "https" if tls else "http"
        result = fetch_remote_image(
            f"{scheme}://images.example:{port}/success",
            Settings(allow_remote_urls=True, max_image_bytes=3),
        )
        assert result == b"abc" and children[0].poll() == 0
        assert seen[0][1]["Host"] == f"images.example:{port}"
        assert (
            not {"Authorization", "Proxy-Authorization", "Cookie"} & seen[0][1].keys()
        )
        assert sni == (["images.example"] if tls else [])
        if (tmp_path / "rss.txt").exists():
            measured = json.loads((tmp_path / "rss.txt").read_text())
            assert 0 < measured["rss_kib"] <= measured["exec_hwm_kib"]


@pytest.mark.parametrize(
    "trusted,hostname", [(False, "images.example"), (True, "wrong.example")]
)
def test_native_worker_rejects_bad_tls_without_url_details(
    monkeypatch, tmp_path, trusted, hostname
):
    with controlled_server(tmp_path, tls=True, trusted=trusted, hostname=hostname) as (
        port,
        certificate,
        seen,
        sni,
    ):
        children = harness(monkeypatch, tmp_path, "network", port, certificate)
        with pytest.raises(RemoteFetchError, match="^Unable to fetch remote image$"):
            fetch_remote_image(
                f"https://images.example:{port}/success?token=test-url-credential",
                Settings(allow_remote_urls=True),
            )
        assert not seen and sni == ["images.example"] and children[0].poll() == 0


def test_native_worker_retains_image_byte_error_contract(monkeypatch, tmp_path):
    with controlled_server(tmp_path) as (port, certificate, _seen, _sni):
        children = harness(monkeypatch, tmp_path, "network", port, certificate)
        with pytest.raises(InputLimitExceeded, match="maximum size"):
            fetch_remote_image(
                f"http://images.example:{port}/success",
                Settings(allow_remote_urls=True, max_image_bytes=2),
            )
        assert children[0].poll() == 0
