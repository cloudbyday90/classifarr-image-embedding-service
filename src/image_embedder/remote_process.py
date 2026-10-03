# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""A parent-owned total fetch budget with termination before ownership release."""

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .config import Settings
from .remote_fetch import RemoteFetchError
from .remote_protocol import (
    MAX_ERROR_BYTES,
    RemoteFetchOptions,
    decode_response,
    encode_request,
)

_BOOTSTRAP = (
    "import sys;sys.path.insert(0,sys.argv[1]);"
    "from image_embedder.remote_worker import main;main()"
)


def _command() -> list[str]:
    # Isolated mode ignores PYTHONPATH, user site packages and the current directory.
    return [
        sys.executable,
        "-I",
        "-c",
        _BOOTSTRAP,
        str(Path(__file__).resolve().parent.parent),
    ]


def _environment() -> dict[str, str]:
    # Windows needs these for interpreter/OS operation. Never inherit service keys,
    # proxy/netrc/CA overrides, Python hooks or dynamic-loader overrides.
    return {
        key: os.environ[key]
        for key in ("SYSTEMROOT", "WINDIR", "TEMP", "TMP")
        if key in os.environ
    }


def fetch_remote_image(
    image_url: str, settings: Settings, *, total_deadline: float | None = None
) -> bytes:
    if not settings.allow_remote_urls:
        raise ValueError("Remote image URLs are disabled")
    deadline = time.monotonic() + settings.remote_fetch_timeout_seconds
    if total_deadline is not None:
        deadline = min(deadline, total_deadline)
    payload = encode_request(
        image_url,
        RemoteFetchOptions(
            settings.allowed_remote_hosts,
            settings.max_image_bytes,
            settings.request_timeout_seconds,
        ),
    )
    process = None
    try:
        # An anonymous file avoids communicate() buffering unchecked stdout in RAM.
        with tempfile.TemporaryFile() as output:
            if time.monotonic() >= deadline:
                raise RemoteFetchError("Remote image fetch timed out")
            process = subprocess.Popen(
                _command(),
                stdin=subprocess.PIPE,
                stdout=output,
                stderr=subprocess.DEVNULL,
                env=_environment(),
                close_fds=True,
            )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(
                    process.args, settings.remote_fetch_timeout_seconds
                )
            process.communicate(input=payload, timeout=remaining)
            if time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(
                    process.args, settings.remote_fetch_timeout_seconds
                )
            if process.returncode != 0:
                raise RemoteFetchError("Unable to fetch remote image")
            limit = max(settings.max_image_bytes + 1, MAX_ERROR_BYTES)
            if output.tell() > limit:
                raise RemoteFetchError("Unable to fetch remote image")
            output.seek(0)
            return decode_response(output.read(limit + 1), settings.max_image_bytes)
    except subprocess.TimeoutExpired:
        raise RemoteFetchError("Remote image fetch timed out") from None
    except OSError:
        raise RemoteFetchError("Unable to fetch remote image") from None
    finally:
        if process is not None:
            try:
                # Popen's context closes pipes and waits even if kill/communicate
                # raises. Cleanup failure must not abandon a still-live owner.
                with process:
                    if process.poll() is None:
                        process.kill()
                    process.communicate()
            except OSError:
                raise RemoteFetchError("Unable to fetch remote image") from None
