# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Batch embedding endpoint: POST /embed-batch."""

import asyncio
import functools

import anyio
from fastapi import APIRouter, Depends, HTTPException, Request, Response

from ..embedder import BatchItem, ImageEmbedder
from ..models import (
    EmbedBatchItemResult,
    EmbedBatchRequest,
    EmbedBatchResponse,
)
from ..queue import EmbedQueue, QueueFullError, QueueWaitTimeoutError
from .embed import _queue_headers


def make_router(limiter, rate_limit_embed: str, auth) -> APIRouter:
    router = APIRouter()

    @router.post("/embed-batch", response_model=EmbedBatchResponse, dependencies=[Depends(auth)])
    @limiter.limit(rate_limit_embed)
    async def embed_batch_endpoint(request: Request, payload: EmbedBatchRequest, response: Response):
        logger = request.app.state.logger
        embedder_instance: ImageEmbedder = request.app.state.embedder
        queue: EmbedQueue = request.app.state.queue
        settings = request.app.state.settings

        max_items: int = settings.embed_batch_api_max_items
        if len(payload.items) > max_items:
            raise HTTPException(
                status_code=413,
                detail=f"Batch exceeds maximum of {max_items} items",
            )

        spec = embedder_instance.resolve_model(payload.model)
        if payload.image_size is not None and payload.image_size != spec.image_size:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"image_size={payload.image_size} is not supported for {spec.name}; "
                    f"this model only accepts image_size={spec.image_size}"
                ),
            )
        target_size = spec.image_size

        batch_items = [
            BatchItem(
                image_url=item.image_url,
                image_base64=item.image_base64,
                normalize=payload.normalize,
            )
            for item in payload.items
        ]

        # Only hit the embedder + queue when there is something to embed.
        embed_results: list = []
        if batch_items:
            acquired = False
            shared = False

            async def _do_embed():
                nonlocal acquired, shared
                try:
                    await queue.acquire()
                    acquired = True
                    await queue.acquire_shared()
                    shared = True
                    return await anyio.to_thread.run_sync(
                        functools.partial(
                            embedder_instance.embed_batch,
                            spec,
                            target_size,
                            batch_items,
                        )
                    )
                finally:
                    if shared:
                        await queue.release_shared()
                    if acquired:
                        await queue.release()

            try:
                embed_results = await asyncio.wait_for(
                    _do_embed(),
                    timeout=settings.request_timeout_seconds,
                )
                if len(embed_results) != len(batch_items):
                    raise RuntimeError(
                        "embed_batch returned "
                        f"{len(embed_results)} results for {len(batch_items)} items"
                    )
            except QueueFullError as exc:
                logger.warning(f"Batch queue full: {exc}")
                raise HTTPException(
                    status_code=429,
                    detail=str(exc),
                    headers=_queue_headers(queue, retry_after_seconds=1),
                ) from exc
            except QueueWaitTimeoutError as exc:
                logger.warning(f"Batch queue wait timeout: {exc}")
                raise HTTPException(
                    status_code=504,
                    detail=str(exc),
                    headers=_queue_headers(queue),
                ) from exc
            except asyncio.TimeoutError as exc:
                logger.warning(
                    f"Batch embedding timed out after {settings.request_timeout_seconds}s"
                )
                raise HTTPException(
                    status_code=504,
                    detail=f"Embedding request timed out after {settings.request_timeout_seconds}s",
                    headers=_queue_headers(queue),
                ) from exc
            except Exception as exc:
                logger.exception(f"Batch embedding error: {exc}")
                raise HTTPException(status_code=500, detail="Internal server error") from exc

        # Map embed results into the final ordered result list.
        results: list[EmbedBatchItemResult] = []
        for i, outcome in enumerate(embed_results):
            if isinstance(outcome, Exception):
                results.append(EmbedBatchItemResult(index=i, status="error", error=str(outcome)))
            else:
                embedding, dims, _provider, _model_name, _img_size = outcome
                results.append(
                    EmbedBatchItemResult(
                        index=i,
                        status="ok",
                        embedding=embedding,
                        dims=dims,
                        model=spec.name,
                        image_size=target_size,
                    )
                )

        succeeded = sum(1 for r in results if r.status == "ok")
        failed = len(results) - succeeded

        for k, v in _queue_headers(queue).items():
            response.headers[k] = v

        return EmbedBatchResponse(
            model=spec.name,
            image_size=target_size,
            total=len(results),
            succeeded=succeeded,
            failed=failed,
            results=results,
        )

    return router
