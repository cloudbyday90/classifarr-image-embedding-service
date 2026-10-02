# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Application-owned admission for complete HTTP request lifetimes."""

from dataclasses import dataclass
from threading import Lock


@dataclass(frozen=True, slots=True)
class IngressStats:
    active: int
    maximum: int
    rejected: int


class IngressAdmission:
    def __init__(self, maximum: int) -> None:
        if type(maximum) is not int or maximum <= 0:
            raise ValueError("HTTP ingress maximum must be a positive integer")
        self._maximum = maximum
        self._active = 0
        self._rejected = 0
        self._lock = Lock()

    def try_acquire(self) -> bool:
        # No await between checking and reserving; no waiting requests/payloads.
        with self._lock:
            if self._active >= self._maximum:
                self._rejected += 1
                return False
            self._active += 1
            return True

    def release(self) -> None:
        with self._lock:
            if self._active == 0:
                raise RuntimeError("HTTP ingress lease released without an owner")
            self._active -= 1

    def stats(self) -> IngressStats:
        with self._lock:
            return IngressStats(self._active, self._maximum, self._rejected)
