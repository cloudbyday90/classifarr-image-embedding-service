# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

from fastapi import FastAPI, HTTPException, Response
import anyio

from .config import Settings
from .embedder import ImageEmbedder
from . import __version__
from .models import EmbedImageRequest, EmbedImageResponse, HealthResponse, ModelInfo
from .queue import EmbedQueue, QueueFullError, QueueWaitTimeoutError


def create_app(embedder: ImageEmbedder | None = None) -> FastAPI:
    settings = Settings()
    app = FastAPI(title="Classifarr Image Embedding Service", version=__version__)

    app.state.embedder = embedder or ImageEmbedder(settings=settings)
    app.state.queue = EmbedQueue(
        concurrency=settings.embed_concurrency,
        max_queue=settings.embed_max_queue,
        max_wait_seconds=settings.embed_max_wait_seconds,
    )

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

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        queue: EmbedQueue = app.state.queue
        return HealthResponse(
            status="ok",
            provider="local",
            default_model=settings.default_model,
            queue=queue.stats().__dict__,
        )

    @app.get("/models", response_model=list[ModelInfo])
    def list_models():
        embedder_instance: ImageEmbedder = app.state.embedder
        return [
            ModelInfo(
                id=spec.name,
                name=spec.name,
                dims=spec.dims,
                image_size=spec.image_size
            )
            for spec in embedder_instance.list_models()
        ]

    @app.post("/embed-image", response_model=EmbedImageResponse)
    async def embed_image(payload: EmbedImageRequest, response: Response):
        embedder_instance: ImageEmbedder = app.state.embedder
        queue: EmbedQueue = app.state.queue
        if not payload.image_url and not payload.image_base64:
            raise HTTPException(
                status_code=400,
                detail="image_url or image_base64 is required"
            )
        try:
                # Embedding work is CPU/GPU bound (torch). Use:
                # - queue slot to avoid contention
                # - shared lock to ensure future exclusive ops (model load/update) don't interrupt active embeds
            await queue.acquire()
            await queue.acquire_shared()
            try:
                embedding, dims, provider, model_name, image_size = await anyio.to_thread.run_sync(
                    embedder_instance.embed,
                    payload.image_url,
                    payload.image_base64,
                    payload.model,
                    payload.normalize,
                    payload.image_size,
                )
            finally:
                await queue.release_shared()
                await queue.release()
        except QueueFullError as exc:
            # Busy: tell caller to retry later.
            raise HTTPException(
                status_code=429,
                detail=str(exc),
                headers=_queue_headers(queue, retry_after_seconds=1),
            ) from exc
        except QueueWaitTimeoutError as exc:
            # Waiting too long for a slot.
            raise HTTPException(
                status_code=504,
                detail=str(exc),
                headers=_queue_headers(queue),
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        for k, v in _queue_headers(queue).items():
            response.headers[k] = v

        return EmbedImageResponse(
            embedding=embedding,
            dims=dims,
            provider=provider,
            model=model_name,
            image_size=image_size
        )

    return app


app = create_app()
