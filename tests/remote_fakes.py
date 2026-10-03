# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Per-test transport seams that keep actual destination validation enabled."""

import socket

from image_embedder import remote_fetch


def dns_answers(monkeypatch, *addresses):
    calls = []

    def resolve(host, port, **kwargs):
        calls.append((host, port))
        return [
            (
                socket.AF_INET6 if ":" in address else socket.AF_INET,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "",
                (address, port),
            )
            for address in addresses
        ]

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    return calls


class RemoteResponse:
    def __init__(self, status=200, headers=None, chunks=(b"abc",), error=None):
        self.status = status
        self.headers = headers or {}
        self.chunks = chunks
        self.error = error
        self.closed = False
        self.streamed = False

    def stream(self, *, amt, decode_content):
        assert amt == 8192 and decode_content
        self.streamed = True
        yield from self.chunks
        if self.error is not None:
            raise self.error

    def close(self):
        self.closed = True


class RemoteTransport:
    def __init__(self, monkeypatch, *outcomes):
        self.outcomes = iter(outcomes)
        self.calls = []
        self.closed = []
        monkeypatch.setattr(remote_fetch, "_pool", self.pool)
        monkeypatch.setattr("image_embedder.embedder.fetch_remote_image", remote_fetch.fetch_remote_image)

    def pool(self, destination, address, timeout):
        owner = self

        class Pool:
            def urlopen(self, method, target, **kwargs):
                owner.calls.append(
                    (destination, address, timeout, method, target, kwargs)
                )
                outcome = next(owner.outcomes)
                if isinstance(outcome, Exception):
                    raise outcome
                return outcome

            def close(self):
                owner.closed.append(address)

        return Pool()
