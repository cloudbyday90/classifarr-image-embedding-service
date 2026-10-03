# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Application factory — wires settings, embedder, queue, lifecycle, and routes together."""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.middleware.body_limit import RequestBodyLimitMiddleware

from . import __version__
from .batch import BatchWindow
from .config import Settings
from .embedder import ImageEmbedder
from .execution import InferenceExecutor
from .ingress import IngressAdmission
from .ingress_middleware import IngressAdmissionMiddleware
from .lifecycle import make_lifespan
from .logging_config import get_logger, setup_logging
from .queue import EmbedQueue
from .rate_limits import make_limiter
from .request_body_deadline import RequestBodyDeadlineMiddleware
from .routes import admin as admin_routes
from .routes import batch as batch_routes
from .routes import embed as embed_routes
from .routes import health as health_routes
from .routes import models as models_routes
from .security import make_auth_dependency


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
    executor = InferenceExecutor(queue)
    ingress = IngressAdmission(settings.max_http_requests)

    limiter = make_limiter(settings)
    auth = make_auth_dependency(settings)

    batch_window: BatchWindow | None = (
        BatchWindow(
            embedder_instance,
            queue,
            batch_window_ms=settings.embed_batch_window_ms,
            batch_max_size=settings.embed_batch_max_size,
            executor=executor,
        )
        if settings.embed_batch_window_ms > 0
        else None
    )

    lifespan = make_lifespan(
        embedder_instance, settings, logger, batch_window=batch_window, executor=executor
    )

    app = FastAPI(
        title="Classifarr Image Embedding Service",
        version=__version__,
        lifespan=lifespan,
    )

    app.state.limiter = limiter
    app.state.embedder = embedder_instance
    app.state.queue = queue
    app.state.executor = executor
    app.state.ingress = ingress
    app.state.batch_window = batch_window
    app.state.settings = settings
    app.state.logger = logger

    # Keep the receive-limit exception inside SlowAPI's BaseHTTP wrapper, so
    # FastAPI preserves HTTP 413 instead of translating an exception group to 400.
    app.add_middleware(RequestBodyLimitMiddleware, max_body_size=settings.max_request_body_bytes)
    app.add_middleware(SlowAPIMiddleware)
    app.add_middleware(RequestBodyDeadlineMiddleware, timeout_seconds=settings.request_body_timeout_seconds)
    app.add_middleware(IngressAdmissionMiddleware, admission=ingress)

    def rate_limit_handler(request: Request, exc: Exception):
        if not isinstance(exc, RateLimitExceeded):
            raise exc
        return _rate_limit_exceeded_handler(request, exc)

    app.add_exception_handler(RateLimitExceeded, rate_limit_handler)

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
    app.include_router(admin_routes.make_router(make_auth_dependency(settings, always_required=True)))
    app.include_router(embed_routes.make_router(limiter, settings.rate_limit_embed, auth))
    app.include_router(batch_routes.make_router(limiter, settings.rate_limit_embed, auth))

    return app


app = create_app()
