# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Event-controlled workers for execution lifetime tests."""

import asyncio
import threading

from fakes import FakeEmbedder


class ThreadGate:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self._lock = threading.Lock()
        self.calls = 0
        self.active = 0
        self.peak = 0

    def run(self, function, *args):
        with self._lock:
            self.calls += 1
            self.active += 1
            self.peak = max(self.peak, self.active)
        self.started.set()
        try:
            if not self.release.wait(5):
                raise RuntimeError("test worker was not released")
            return function(*args)
        finally:
            with self._lock:
                self.active -= 1

    async def wait_started(self):
        assert await asyncio.to_thread(self.started.wait, 2), "worker never started"


class GatedEmbedder(FakeEmbedder):
    def __init__(self, gate):
        super().__init__()
        self.gate = gate

    def embed(self, *args):
        return self.gate.run(super().embed, *args)

    def embed_batch(self, *args):
        return self.gate.run(super().embed_batch, *args)


async def wait_until(predicate):
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(0)
