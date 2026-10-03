# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Test-only fresh-child mappings; production never accepts a worker command."""

import json
import os
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, sys.argv[1])

from image_embedder import remote_fetch, remote_worker  # noqa: E402

scenario, port, certificate = sys.argv[2:5]
assert (
    not {
        "SERVICE_API_KEY",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NETRC",
        "REQUESTS_CA_BUNDLE",
        "SSL_CERT_FILE",
        "LD_PRELOAD",
    }
    & os.environ.keys()
)
assert "torch" not in sys.modules and "transformers" not in sys.modules

if scenario == "dns":

    def blocked_dns(*_args, **_kwargs):
        time.sleep(60)

    socket.getaddrinfo = blocked_dns
else:
    original_dns = socket.getaddrinfo
    original_connect = socket.socket.connect

    def public_dns(host, requested_port, *args, **kwargs):
        if host == "images.example":
            return [
                (
                    socket.AF_INET,
                    socket.SOCK_STREAM,
                    socket.IPPROTO_TCP,
                    "",
                    ("8.8.8.8", requested_port),
                )
            ]
        assert host == "8.8.8.8"
        return original_dns(host, requested_port, *args, **kwargs)

    def controlled_connect(sock, address):
        assert address == ("8.8.8.8", int(port))
        return original_connect(sock, ("127.0.0.1", int(port)))

    socket.getaddrinfo = public_dns
    socket.socket.connect = controlled_connect
    if certificate:
        remote_fetch.certifi.where = lambda: certificate

remote_worker.main()

if sys.platform == "linux":
    import resource

    status = dict(
        line.split(":", 1)
        for line in Path("/proc/self/status").read_text().splitlines()
    )
    Path(sys.argv[5]).write_text(
        json.dumps(
            {
                "rss_kib": int(status["VmRSS"].split()[0]),
                "exec_hwm_kib": int(status["VmHWM"].split()[0]),
                "rusage_hwm_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            }
        )
    )
