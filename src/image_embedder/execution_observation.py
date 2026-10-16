# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Optional trusted observers; observations never own inference capacity."""

from typing import Literal, Protocol

from .logging_config import get_logger

ExecutionPhase = Literal[
    "admission_started",
    "admitted",
    "read_lock_acquired",
    "dispatched",
    "native_started",
    "native_failed",
    "native_finished",
    "waiter_detached",
    "owner_failed",
    "owner_cancelled",
    "owner_completed",
]
logger = get_logger(__name__)


class ExecutionTrace(Protocol):
    def mark(self, phase: ExecutionPhase) -> None:
        """Observe without blocking; native marks run on the worker thread."""
        ...


class ExecutionObserver(Protocol):
    def begin(self) -> ExecutionTrace | None:
        """Capture the submitting context without callable arguments or results."""
        ...


def begin_observation(observer: ExecutionObserver | None) -> ExecutionTrace | None:
    if observer is not None:
        try:
            return observer.begin()
        except Exception:
            logger.warning("Inference observer could not begin")
    return None


def mark_observation(trace: ExecutionTrace | None, phase: ExecutionPhase) -> None:
    if trace is not None:
        try:
            trace.mark(phase)
        except Exception:
            # Never include private observer state or exception text in logs.
            logger.warning("Inference observer could not record phase")
