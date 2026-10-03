# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Bound private client IPC and own an isolated interpreter until it is reaped."""

import asyncio
import json
import os
import sys
from pathlib import Path

MAX_INPUT_BYTES = 1024 * 1024
MAX_OUTPUT_BYTES = 256 * 1024
_BOOTSTRAP = (
    "import sys;sys.path.insert(0,sys.argv[1]);"
    "from capacity_representative_client import main;main()"
)


class ClientFailed(RuntimeError):
    def __init__(self, report: dict) -> None:
        super().__init__("Representative client failed")
        self.report = report


def client_command() -> list[str]:
    return [
        sys.executable,
        "-I",
        "-c",
        _BOOTSTRAP,
        str(Path(__file__).resolve().parent),
    ]


def client_environment() -> dict[str, str]:
    return {
        name: os.environ[name]
        for name in ("SYSTEMROOT", "WINDIR", "TEMP", "TMP")
        if name in os.environ
    }


async def _exchange(process, encoded: bytes) -> bytes:
    async def write():
        process.stdin.write(encoded)
        await process.stdin.drain()
        process.stdin.close()
        await process.stdin.wait_closed()

    async def read():
        chunks, total = [], 0
        while chunk := await process.stdout.read(65536):
            total += len(chunk)
            if total > MAX_OUTPUT_BYTES:
                raise ValueError("Client output exceeds experiment ceiling")
            chunks.append(chunk)
        return b"".join(chunks)

    async with asyncio.TaskGroup() as group:
        group.create_task(write())
        output = group.create_task(read())
        group.create_task(process.wait())
    return output.result()


async def _reap(spawn) -> None:
    process = await spawn
    if process.returncode is None:
        try:
            process.kill()
        except ProcessLookupError:
            pass
    if process.stdin is not None:
        process.stdin.close()
    # Drain without retaining output: a paused pipe can otherwise delay wait().
    if process.stdout is not None:
        while await process.stdout.read(65536):
            pass
    await process.wait()


async def run_client(message: dict, timeout: float = 300) -> dict:
    encoded = json.dumps(message, allow_nan=False, separators=(",", ":")).encode(
        "ascii"
    )
    if len(encoded) > MAX_INPUT_BYTES:
        raise ValueError("Client input exceeds experiment ceiling")
    if not 0 < timeout <= 300:
        raise ValueError("Client lifetime must fit the experiment ceiling")
    spawn = asyncio.create_task(
        asyncio.create_subprocess_exec(
            *client_command(),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env=client_environment(),
            limit=65536,
        )
    )
    try:
        async with asyncio.timeout(timeout):
            process = await asyncio.shield(spawn)
            output = await _exchange(process, encoded)
            try:
                result = json.loads(output)
            except (ValueError, UnicodeError):
                raise RuntimeError("Invalid representative client report") from None
            if (
                process.returncode != 0
                and isinstance(result, dict)
                and result.get("passed") is False
                and isinstance(result.get("records"), list)
            ):
                raise ClientFailed(
                    {**result, "reaped": True, "returncode": process.returncode}
                )
            if process.returncode != 0:
                raise RuntimeError("Representative client failed")
            if not isinstance(result, dict) or result.get("passed") is not True:
                raise ValueError("Invalid representative client report")
            return {**result, "reaped": True, "returncode": process.returncode}
    finally:
        cleanup = asyncio.create_task(_reap(spawn))
        canceled = False
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                canceled = True
        cleanup.result()
        if canceled:
            raise asyncio.CancelledError
