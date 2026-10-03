# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Run offline production capacity experiments inside a memory-limited Linux container."""

import argparse
import json
import math
import os
import platform
from contextlib import ExitStack
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir

from capacity_metrics import MemoryReader
from capacity_resources import ResourceReader
from capacity_workload import CapacityWorkload, make_payloads

from image_embedder.config import Settings
from image_embedder.embedder import ImageEmbedder
from image_embedder.model_catalog import MODEL_CATALOG


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["cpu", "openvino", "cuda"], required=True)
    parser.add_argument(
        "--models", choices=list(MODEL_CATALOG), nargs="+", default=list(MODEL_CATALOG)
    )
    parser.add_argument(
        "--scenario",
        choices=["serial", "overlap", "detached", "api", "mixed", "socket", "representative"],
        default="serial",
    )
    parser.add_argument(
        "--cold-ir", action="store_true", help="Export into a private temporary cache"
    )
    parser.add_argument("--image-edge", type=positive_int, default=224,
                        help="Square fixture edge; representative uses its fixed image profiles")
    parser.add_argument("--repeats", type=positive_int, default=2)
    parser.add_argument("--clients", type=positive_int, default=2,
                        help="Concurrent HTTP callers in the representative scenario (1..8)")
    parser.add_argument("--batch-sizes", type=positive_int, nargs="+",
                        help="Representative subset of 1/internal/API ceilings; default all")
    parser.add_argument("--sample-interval", type=float, default=0.1)
    parser.add_argument("--output", type=Path, help="Flushed JSONL measurement journal")
    args = parser.parse_args()
    if (
        not math.isfinite(args.sample_interval)
        or not 0.01 <= args.sample_interval <= 10
    ):
        parser.error("Sample interval must be finite and between 0.01 and 10 seconds")
    if platform.system() != "Linux":
        parser.error("Run this probe inside the shipped Linux image")
    if os.getenv("HF_HUB_OFFLINE") != "1" or os.getenv("TRANSFORMERS_OFFLINE") != "1":
        parser.error("Populate verified assets first, then set both offline flags to 1")
    if args.cold_ir and args.backend != "openvino":
        parser.error("--cold-ir requires OpenVINO")
    memory_reader = MemoryReader()
    resource_reader = ResourceReader(Path(gettempdir()))
    cuda = None

    def reader() -> dict:
        return {
            **memory_reader(),
            "resources": resource_reader(),
            "cuda": cuda() if cuda is not None else None,
        }

    memory = reader()
    if not memory["cgroup"] or memory["cgroup"]["limit_bytes"] is None:
        parser.error("A finite cgroup memory limit is required")
    settings = Settings(
        device={"cpu": "cpu", "openvino": "openvino:CPU", "cuda": "cuda"}[args.backend],
        warmup_on_startup=False,
        embed_cache_size=0,
        require_api_key=False,
        cleanup_on_shutdown=False,
        embed_concurrency=1,
        embed_batch_window_ms=0,
        allow_remote_urls=args.scenario in {"mixed", "socket"},
        allowed_remote_hosts=["capacity.example"] if args.scenario in {"mixed", "socket"} else [],
    )
    if args.scenario == "socket":
        from capacity_upload_body import validate_socket_limits

        validate_socket_limits(settings)
    sizes = sorted(
        {1, settings.embed_batch_max_size, settings.embed_batch_api_max_items}
    )
    if min(sizes) <= 0 or max(sizes) > 32:
        parser.error(
            "This bounded experiment supports positive batch ceilings up to 32"
        )
    if args.batch_sizes is not None:
        if args.scenario != "representative" or not set(args.batch_sizes) <= set(sizes):
            parser.error("--batch-sizes requires representative and configured batch ceilings")
        sizes = sorted(set(args.batch_sizes))
    # Square images account for both source pixels and the 224-pixel preprocessing resize.
    if args.scenario != "representative" and max(sizes) * (args.image_edge**2 + 224**2) > settings.max_batch_image_pixels:
        parser.error("Image edge and batch ceilings exceed the aggregate pixel budget")
    fixtures = None
    if args.scenario == "representative":
        from capacity_image_fixtures import (
            codec_versions,
            make_fixtures,
            validate_fixtures,
        )

        fixtures = make_fixtures()
        try:
            validate_fixtures(fixtures, settings, [MODEL_CATALOG[name] for name in args.models],
                              sizes, args.clients, args.repeats)
        except ValueError as error:
            parser.error(str(error))
    with ExitStack() as resources:
        stream = (
            resources.enter_context(args.output.open("w", encoding="utf-8"))
            if args.output
            else None
        )

        def emit(record: dict) -> None:
            encoded = json.dumps(record, sort_keys=True, allow_nan=False)
            print(encoded, flush=True)
            if stream is not None:
                stream.write(encoded + "\n")
                stream.flush()

        original_ir = os.environ.get("OV_MODEL_CACHE")
        if args.cold_ir:
            root = Path(os.environ.get("HF_HOME", "/app/.cache"))
            root.mkdir(parents=True, exist_ok=True)
            os.environ["OV_MODEL_CACHE"] = resources.enter_context(
                TemporaryDirectory(prefix="capacity-ir-", dir=root)
            )
        try:
            import torch

            if args.backend == "cuda":
                from capacity_cuda import CudaReader

                cuda = CudaReader(torch.cuda)

            models = list(dict.fromkeys(args.models))
            runtimes = {
                name: version(name)
                for name in ("torch", "transformers", "pydantic", "pydantic-core", "pillow", "numpy", "httpx", "uvicorn")
            }
            if args.backend == "openvino":
                runtimes["openvino"] = version("openvino")
            emit(
                {
                    "event": "configuration",
                    "backend": args.backend,
                    "scenario": args.scenario,
                    "cold_ir": args.cold_ir,
                    "models": {name: MODEL_CATALOG[name].revision for name in models},
                    "batch_sizes": sizes,
                    "image_edge": args.image_edge if fixtures is None else None,
                    "repeats": args.repeats,
                    "representative_clients": args.clients if fixtures is not None else None,
                    "representative_fixtures": [fixture.metadata for fixture in fixtures]
                    if fixtures is not None else None,
                    "codec_versions": codec_versions() if fixtures is not None else None,
                    "sample_interval_seconds": args.sample_interval,
                    "torch_threads": torch.get_num_threads(),
                    "torch_interop_threads": torch.get_num_interop_threads(),
                    "runtimes": runtimes,
                    "worker_processes": 1,
                    "inference_concurrency": 1,
                    "embedding_timeout_seconds": settings.embedding_timeout_seconds,
                    "remote_hop_timeout_seconds": settings.request_timeout_seconds,
                    "load_pressure_concurrency": len(models)
                    if args.scenario == "overlap"
                    else 1,
                    "memory": reader(),
                }
            )
            workload = CapacityWorkload(
                ImageEmbedder(settings),
                models,
                sizes,
                make_payloads(args.image_edge) if fixtures is None else [],
                reader,
                emit,
                args.sample_interval,
                args.repeats,
                expected_device={"cpu": "cpu", "openvino": "ov:CPU", "cuda": "cuda"}[
                    args.backend
                ],
                cuda=cuda,
                clients=args.clients,
            )
            workload.run(args.scenario)
            emit({"event": "complete", "passed": True, "memory": reader()})
        except Exception as error:
            emit(
                {
                    "event": "failed",
                    "error_type": type(error).__name__,
                    "memory": reader(),
                }
            )
            raise
        finally:
            if original_ir is None:
                os.environ.pop("OV_MODEL_CACHE", None)
            else:
                os.environ["OV_MODEL_CACHE"] = original_ir


if __name__ == "__main__":
    main()
