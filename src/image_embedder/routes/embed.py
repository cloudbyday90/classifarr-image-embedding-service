# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Embedding endpoint: POST /embed-image."""

import asyncio

import anyio
from fastapi import APIRouter, Depends, HTTPException, Request, Response

from ..batch import EmbedJob
from ..embedder import ImageEmbedder
from ..models import EmbedImageRequest, EmbedImageResponse
from ..queue import EmbedQueue, QueueFullError, QueueWaitTimeoutError


def _queue_headers(queue: EmbedQueue, *, retry_after_seconds: int | None = None) -> dict[str, str]:
    stats = queue.stats()
    headers = {
        "X-Queue-Concurrency": str(stats.concurrency),
        "X-Queue-In-Flight": str(stats.in_flight),
        "X-Queue-Waiting": str(stats.waiting),
        "X-Queue-Max-Queue": str(stats.max_queue),
        "X-Queue-Max-Wait-Seconds": str(stats.max_wait_seconds),
    }
    if retry_after_seconds is not None:
        headers["Retry-After"] = str(max(1, int(retry_after_seconds)))
    return headers


def make_router(limiter, rate_limit_embed: str, auth) -> APIRouter:
    router = APIRouter()

    @router.post("/embed-image", response_model=EmbedImageResponse, dependencies=[Depends(auth)])
    @limiter.limit(rate_limit_embed)
    async def embed_image(request: Request, payload: EmbedImageRequest, response: Response):
        logger = request.app.state.logger
        embedder_instance: ImageEmbedder = request.app.state.embedder
        queue: EmbedQueue = request.app.state.queue
        settings = request.app.state.settings

        if not payload.image_url and not payload.image_base64:
            raise HTTPException(
                status_code=400,
                detail="image_url or image_base64 is required",
            )

        batch_window = getattr(request.app.state, "batch_window", None)

        async def _do_embed():
            if batch_window is not None:
                job = EmbedJob(
                    image_url=payload.image_url,
                    image_base64=payload.image_base64,
                    model=payload.model,
                    normalize=payload.normalize,
                    image_size=payload.image_size,
                )
                return await batch_window.submit(job)

            # Standard single-request path.
            acquired = False
            shared = False
            try:
                await queue.acquire()
                acquired = True
                await queue.acquire_shared()
                shared = True
                return await anyio.to_thread.run_sync(  # type: ignore[union-attr]
                    embedder_instance.embed,
                    payload.image_url,
                    payload.image_base64,
                    payload.model,
                    payload.normalize,
                    payload.image_size,
                )
            finally:
                if shared:
                    await queue.release_shared()
                if acquired:
                    await queue.release()

        try:
            embedding, dims, provider, model_name, image_size = await asyncio.wait_for(
                _do_embed(),
                timeout=settings.request_timeout_seconds,
            )
        except QueueFullError as exc:
            logger.warning(f"Queue full: {exc}")
            raise HTTPException(
                status_code=429,
                detail=str(exc),
                headers=_queue_headers(queue, retry_after_seconds=1),
            ) from exc
        except QueueWaitTimeoutError as exc:
            logger.warning(f"Queue wait timeout: {exc}")
            raise HTTPException(
                status_code=504,
                detail=str(exc),
                headers=_queue_headers(queue),
            ) from exc
        except asyncio.TimeoutError as exc:
            logger.warning(
                f"Embedding request timed out after {settings.request_timeout_seconds}s"
            )
            raise HTTPException(
                status_code=504,
                detail=f"Embedding request timed out after {settings.request_timeout_seconds}s",
                headers=_queue_headers(queue),
            ) from exc
        except ValueError as exc:
            logger.warning(f"Validation error: {exc}")
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception(f"Embedding error: {exc}")
            raise HTTPException(status_code=500, detail="Internal server error") from exc

        for k, v in _queue_headers(queue).items():
            response.headers[k] = v

        return EmbedImageResponse(
            embedding=embedding,
            dims=dims,
            provider=provider,
            model=model_name,
            image_size=image_size,
        )

    return router
