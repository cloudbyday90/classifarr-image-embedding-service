# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Application factory — wires settings, embedder, queue, lifecycle, and routes together."""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from . import __version__
from .batch import BatchWindow
from .config import Settings
from .embedder import ImageEmbedder
from .lifecycle import make_lifespan
from .logging_config import get_logger, setup_logging
from .queue import EmbedQueue
from .routes import admin as admin_routes
from .routes import batch as batch_routes
from .routes import embed as embed_routes
from .routes import health as health_routes
from .routes import models as models_routes
from .security import make_auth_dependency, make_limiter


def create_app(embedder: ImageEmbedder | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()

    logger = setup_logging(
        level=settings.log_level,
        log_file=settings.log_file,
        max_bytes=settings.log_max_bytes,
        backup_count=settings.log_backup_count,
        json_format=settings.log_json_format,
    )
    logger.info(f"Starting Classifarr Image Embedding Service v{__version__}")

    embedder_instance = embedder or ImageEmbedder(settings=settings)
    queue = EmbedQueue(
        concurrency=settings.embed_concurrency,
        max_queue=settings.embed_max_queue,
        max_wait_seconds=settings.embed_max_wait_seconds,
    )

    limiter = make_limiter(settings)
    auth = make_auth_dependency(settings)

    batch_window: BatchWindow | None = (
        BatchWindow(
            embedder_instance,
            queue,
            batch_window_ms=settings.embed_batch_window_ms,
            batch_max_size=settings.embed_batch_max_size,
        )
        if settings.embed_batch_window_ms > 0
        else None
    )

    lifespan = make_lifespan(embedder_instance, settings, logger, batch_window=batch_window)

    app = FastAPI(
        title="Classifarr Image Embedding Service",
        version=__version__,
        lifespan=lifespan,
    )

    app.state.limiter = limiter
    app.state.embedder = embedder_instance
    app.state.queue = queue
    app.state.batch_window = batch_window
    app.state.settings = settings
    app.state.logger = logger

    app.add_middleware(SlowAPIMiddleware)
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        _logger = get_logger("image_embedder.errors")
        _logger.exception(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )

    app.include_router(health_routes.make_router(limiter, settings.rate_limit_health))
    app.include_router(models_routes.make_router(auth))
    app.include_router(admin_routes.make_router(auth))
    app.include_router(embed_routes.make_router(limiter, settings.rate_limit_embed, auth))
    app.include_router(batch_routes.make_router(limiter, settings.rate_limit_embed, auth))

    return app


app = create_app()
