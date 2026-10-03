# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Probe-only loopback transport; production launch/policy has no fixture option."""

import base64
import subprocess
import sys
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from image_embedder import remote_process

# Resolve the fixed authority to a public address before mapping only its approved
# numeric connection to the local fixture. This is an experiment, not SSRF coverage.
_BOOTSTRAP = """
import socket, sys, time
sys.path.insert(0, sys.argv[1])
from image_embedder import remote_fetch, remote_worker
resolve = socket.getaddrinfo
def addresses(host, port, *args, **options):
    if host == 'capacity.example':
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('93.184.216.34', port))]
    if host != '127.0.0.1':
        raise ValueError('Unexpected fixture connection')
    return resolve(host, port, *args, **options)
socket.getaddrinfo = addresses
def pool(destination, address, timeout):
    if destination.host != 'capacity.example' or address != '93.184.216.34' or destination.scheme != 'http':
        raise ValueError('Unexpected fixture destination')
    return remote_fetch.urllib3.HTTPConnectionPool('127.0.0.1', port=int(sys.argv[2]), timeout=timeout)
remote_fetch._pool = pool
remote_worker.main()
time.sleep(0.3)
"""


@contextmanager
def remote_fixture(payloads: list[str], body_bytes: int):
    bodies = []
    for payload in payloads:
        data = base64.b64decode(payload, validate=True)
        if len(data) > body_bytes:
            raise ValueError("Fixture PNG exceeds remote byte pressure")
        bodies.append(data + b"\0" * (body_bytes - len(data)))
    started = threading.Event()
    children = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path not in ("/0", "/1"):
                self.send_error(404)
                return
            body = bodies[int(self.path[1:])]
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except OSError:
                pass

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    original_popen = subprocess.Popen

    def launch(*args, **kwargs):
        child = original_popen(*args, **kwargs)
        children.append(child)
        started.set()
        return child

    command = [
        sys.executable,
        "-I",
        "-c",
        _BOOTSTRAP,
        str(Path(remote_process.__file__).resolve().parent.parent),
        str(server.server_port),
    ]
    thread.start()
    try:
        with (
            patch.object(remote_process, "_command", return_value=command),
            patch.object(remote_process.subprocess, "Popen", side_effect=launch),
        ):
            yield started, children
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        for child in children:
            if child.poll() is None:
                child.kill()
            child.communicate()
