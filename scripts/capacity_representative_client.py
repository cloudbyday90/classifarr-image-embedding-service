# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Separate HTTP client: encoded inputs, bounded responses and native vector parity."""

import asyncio
import json
import os
import sys
import time

import httpx
import numpy as np
from capacity_client_process import MAX_INPUT_BYTES, MAX_OUTPUT_BYTES
from capacity_image_fixtures import batch_body, make_fixtures
from capacity_metrics import MemoryReader

MAX_RESPONSE_BYTES = 2 * 1024 * 1024


def validate_result(
    data, model: dict, references: list, size: int, normalize: bool
) -> None:
    if (
        len(data["results"]) != size
        or data["total"] != size
        or data["succeeded"] != size
        or data["failed"] != 0
        or data["model"] != model["name"]
        or data["image_size"] != model["image_size"]
    ):
        raise AssertionError("Representative response count changed")
    for index, result in enumerate(data["results"]):
        if (
            result.get("error") is not None
            or result["status"] != "ok"
            or result["index"] != index
            or result["dims"] != model["dims"]
            or result["model"] != model["name"]
            or result["image_size"] != model["image_size"]
        ):
            raise AssertionError("Representative response metadata changed")
        expected = np.asarray(references[index % 2], dtype=float)
        actual = np.asarray(result["embedding"], dtype=float)
        if expected.shape != (model["dims"],) or actual.shape != expected.shape:
            raise AssertionError("Representative response dimensions changed")
        if not np.isfinite(actual).all() or not np.isfinite(expected).all():
            raise AssertionError("Representative response contains nonfinite vectors")
        if normalize:
            expected = expected / max(float(np.linalg.norm(expected)), 1e-12)
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)


async def request_case(
    client, model, fixture, size, repeat, caller, normalize, references, records
):
    body = batch_body(fixture, model["name"], size, normalize)
    started = time.monotonic()
    received = bytearray()
    record = {
        "label": f"{model['name']}/{fixture.name}/{size}/{repeat}/{caller}",
        "model": model["name"],
        "fixture": fixture.name,
        "batch_size": size,
        "caller": caller,
        "repeat": repeat,
        "normalize": normalize,
        "request_bytes": len(body),
        "status": None,
        "passed": False,
    }
    try:
        async with client.stream(
            "POST",
            "/embed-batch",
            content=body,
            headers={"X-Capacity-Id": record["label"]},
        ) as response:
            record["status"] = response.status_code
            if response.status_code != 200:
                raise AssertionError("Representative request was refused")
            async for chunk in response.aiter_bytes(65536):
                if len(received) + len(chunk) > MAX_RESPONSE_BYTES:
                    raise ValueError("Representative response exceeds ceiling")
                received.extend(chunk)
        validate_result(json.loads(received), model, references, size, normalize)
        record["passed"] = True
    except (Exception, asyncio.CancelledError) as error:
        record["error_type"] = type(error).__name__
        raise
    finally:
        record["response_bytes"] = len(received)
        record["elapsed_seconds"] = time.monotonic() - started
        records.append(record)


def client_report(passed: bool, records: list) -> dict:
    memory = MemoryReader()()
    return {
        "passed": passed,
        "pid": os.getpid(),
        "records": records,
        "process_rss_bytes": memory["process_rss_bytes"],
        "process_peak_rss_bytes": memory["process_peak_rss_bytes"],
    }


async def run_cases(config: dict) -> dict:
    port, clients, repeats, sizes = (
        config[name] for name in ("port", "clients", "repeats", "sizes")
    )
    if (
        type(port) is not int
        or not 1 <= port <= 65535
        or type(clients) is not int
        or not 1 <= clients <= 8
        or type(repeats) is not int
        or not 1 <= repeats <= 2
        or not sizes
        or len(sizes) > 3
        or any(type(size) is not int or not 1 <= size <= 32 for size in sizes)
        or not 1 <= len(config["models"]) <= 2
    ):
        raise ValueError("Invalid representative client configuration")
    fixtures = make_fixtures()
    if [fixture.metadata for fixture in fixtures] != config["fixtures"]:
        raise AssertionError("Client encoded fixtures differ from native references")
    records = []
    async with httpx.AsyncClient(
        base_url=f"http://127.0.0.1:{port}",
        trust_env=False,
        timeout=120,
        headers={"X-Api-Key": config["key"], "Content-Type": "application/json"},
        limits=httpx.Limits(max_connections=clients, max_keepalive_connections=clients),
    ) as client:
        for model in config["models"]:
            for fixture_index, fixture in enumerate(fixtures):
                for size in sizes:
                    for repeat in range(repeats):
                        try:
                            async with asyncio.TaskGroup() as group:
                                for caller in range(clients):
                                    group.create_task(
                                        request_case(
                                            client,
                                            model,
                                            fixture,
                                            size,
                                            repeat,
                                            caller,
                                            bool((fixture_index + repeat + caller) % 2),
                                            config["references"][model["name"]][
                                                fixture.name
                                            ],
                                            records,
                                        )
                                    )
                        except ExceptionGroup:
                            return client_report(False, records)
    return client_report(True, records)


def main() -> None:
    try:
        encoded = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
        if len(encoded) > MAX_INPUT_BYTES:
            raise ValueError("Representative input exceeds ceiling")
        result = asyncio.run(run_cases(json.loads(encoded)))
        output = json.dumps(result, allow_nan=False, separators=(",", ":"))
        if len(output.encode("ascii")) > MAX_OUTPUT_BYTES:
            raise ValueError("Representative report exceeds ceiling")
    except Exception as error:
        print(
            json.dumps({"passed": False, "error_type": type(error).__name__}),
            flush=True,
        )
        raise SystemExit(1) from None
    print(output, flush=True)
    if result["passed"] is not True:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
