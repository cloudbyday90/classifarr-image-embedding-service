# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Cooperative stop requests belong to one inference owner, not its model."""

from contextlib import contextmanager
from contextvars import ContextVar
from threading import Event


class InferenceCancelled(Exception):
    """No waiter remains; undispatched synchronous work may stop safely."""


_current: ContextVar[Event | None] = ContextVar("inference_stop", default=None)


@contextmanager
def cancellation_scope(stop: Event):
    token = _current.set(stop)
    try:
        yield
    finally:
        _current.reset(token)


def cancellation_checkpoint() -> None:
    stop = _current.get()
    if stop is not None and stop.is_set():
        raise InferenceCancelled("Inference no longer has a waiter")
