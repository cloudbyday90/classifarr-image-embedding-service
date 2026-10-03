# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Representative native HTTP bursts without HTTP-client allocations in server RSS."""

import asyncio
import secrets
from dataclasses import asdict, replace

from capacity_client_process import ClientFailed, run_client
from capacity_image_fixtures import codec_versions, make_fixtures, validate_fixtures
from capacity_receive_observer import ReceiveObserver
from capacity_socket_server import header_only_status, socket_server, wait_until


async def check_representative(workload, clients: int) -> dict:
    from image_embedder.main import create_app

    settings = workload.embedder.settings
    fixtures = make_fixtures()
    models = [workload.embedder.resolve_model(name) for name in workload.models]
    validate_fixtures(
        fixtures, settings, models, workload.sizes, clients, workload.repeats
    )
    references = {}
    for model in models:
        references[model.name] = {}
        for fixture in fixtures:
            references[model.name][fixture.name] = [
                (
                    await asyncio.to_thread(
                        workload.embedder.embed,
                        None,
                        payload,
                        model.name,
                        False,
                        model.image_size,
                    )
                )[0]
                for payload in fixture.payloads
            ]
    # Calibration-only quota allows finite bursts; shipped settings are unchanged.
    protected = replace(
        settings,
        require_api_key=True,
        service_api_key=secrets.token_urlsafe(24),
        rate_limit_embed="1000/minute",
        allow_remote_urls=False,
        allowed_remote_hosts=[],
    )
    app = create_app(workload.embedder, protected)
    observer = ReceiveObserver(app)
    async with socket_server(observer, protected) as port:
        unauthorized = await header_only_status(port, 1024, None, "unauthenticated")
        await wait_until(lambda: observer.records["unauthenticated"].completed)
        if unauthorized != 401 or observer.records["unauthenticated"].bytes_received:
            raise AssertionError("Representative authentication consumed body")
        failure = None
        try:
            report = await run_client(
                {
                    "port": port,
                    "key": protected.service_api_key,
                    "clients": clients,
                    "repeats": workload.repeats,
                    "sizes": workload.sizes,
                    "models": [
                        {
                            "name": model.name,
                            "dims": model.dims,
                            "image_size": model.image_size,
                        }
                        for model in models
                    ],
                    "fixtures": [fixture.metadata for fixture in fixtures],
                    "references": references,
                }
            )
        except ClientFailed as error:
            report, failure = error.report, error
        await wait_until(
            lambda: all(record.completed for record in observer.records.values())
        )
        records = report["records"]
        expected_count = (
            len(models)
            * len(fixtures)
            * len(workload.sizes)
            * clients
            * workload.repeats
        )
        if failure is None and (
            len(records) != expected_count
            or len(observer.records) != expected_count + 1
        ):
            raise AssertionError("Representative request count changed")
        for record in records:
            observed = observer.records.get(record["label"])
            if record["passed"] and (
                observed is None
                or observed.status != 200
                or observed.bytes_received != record["request_bytes"]
            ):
                raise AssertionError(
                    "Representative receipt differs from client report"
                )
            record["server_received_bytes"] = (
                observed.bytes_received if observed else None
            )
            record["server_status"] = observed.status if observed else None
    ingress, queue = asdict(app.state.ingress.stats()), asdict(app.state.queue.stats())
    if ingress["active"] or any(
        queue[name] for name in ("in_flight", "waiting", "rw_readers")
    ):
        raise AssertionError("Representative owners did not settle")
    result = {
        "transport": "direct-loopback-http1",
        "clients": clients,
        "fixtures": [fixture.metadata for fixture in fixtures],
        "codec_versions": codec_versions(),
        "client": report,
        "settled_ingress": ingress,
        "settled_queue": queue,
        "unauthenticated_status": unauthorized,
        "unauthenticated_received_bytes": 0,
        "calibration_rate_limit": protected.rate_limit_embed,
        "fixture_scope": "separate HTTP client process; shared container/cgroup; server RSS includes references/sampler",
    }
    if failure is not None:
        workload.emit({"event": "representative_inputs", "result": result})
        raise failure
    return result
