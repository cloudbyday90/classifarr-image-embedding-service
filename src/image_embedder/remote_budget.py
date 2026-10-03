# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""One lazy fetch-phase deadline per sequential embedding batch."""

import time

from .config import Settings
from .remote_fetch import RemoteFetchError
from .remote_process import fetch_remote_image


class RemoteBatchBudget:
    """Share elapsed time without changing settings or inference ownership."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._deadline: float | None = None

    def fetch(self, image_url: str) -> bytes:
        now = time.monotonic()
        if self._deadline is None:
            self._deadline = now + self._settings.remote_batch_fetch_timeout_seconds
        deadline = self._deadline
        if now >= deadline:
            raise RemoteFetchError("Remote batch fetch timed out")
        try:
            payload = fetch_remote_image(
                image_url, self._settings, total_deadline=deadline
            )
        except RemoteFetchError:
            # The supervisor has already killed/reaped its child before raising.
            if time.monotonic() >= deadline:
                raise RemoteFetchError("Remote batch fetch timed out") from None
            raise
        # Includes result validation and cleanup; late bytes never enter the cache.
        if time.monotonic() >= deadline:
            raise RemoteFetchError("Remote batch fetch timed out")
        return payload
