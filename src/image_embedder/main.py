# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import asyncio
import signal
from contextlib import asynccontextmanager

import anyio
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from .config import Settings
from .embedder import ImageEmbedder
from .logging_config import setup_logging, get_logger
from .memory import cleanup_gpu_memory, force_cleanup, get_memory_usage, check_memory_health
from . import __version__
from .models import (
    EmbedImageRequest, EmbedImageResponse, HealthResponse, ModelInfo,
    ReadyResponse, DeviceInfo, ModelStatus, MemoryInfo, CleanupResponse,
)
from .queue import EmbedQueue, QueueFullError, QueueWaitTimeoutError


def create_app(embedder: ImageEmbedder | None = None) -> FastAPI:
    settings = Settings()
    
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
    
    shutdown_event = asyncio.Event()
    memory_cleanup_task: asyncio.Task | None = None

    async def memory_cleanup_loop():
        while not shutdown_event.is_set():
            try:
                await asyncio.sleep(settings.memory_cleanup_interval_seconds)
                if not shutdown_event.is_set():
                    is_healthy, issues = check_memory_health(
                        max_process_mb=settings.max_process_memory_mb,
                        max_gpu_mb=settings.max_gpu_memory_mb,
                    )
                    if not is_healthy:
                        logger.warning(f"Memory health check failed: {issues}")
                        result = cleanup_gpu_memory()
                        logger.info(f"Periodic cleanup: {result}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in memory cleanup loop: {e}")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal memory_cleanup_task
        
        loop = asyncio.get_event_loop()
        
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown")
            shutdown_event.set()
        
        try:
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(sig, lambda s=sig: signal_handler(s, None))
        except (NotImplementedError, RuntimeError):
            pass
        
        if settings.warmup_on_startup:
            logger.info("Warming up default model...")
            try:
                await anyio.to_thread.run_sync(embedder_instance.warmup)  # type: ignore[union-attr]
                logger.info("Model warmup complete")
            except Exception as e:
                logger.error(f"Model warmup failed: {e}")
                raise
        
        memory_cleanup_task = asyncio.create_task(memory_cleanup_loop())
        logger.info("Started memory cleanup background task")
        
        yield
        
        logger.info("Shutdown initiated")
        shutdown_event.set()
        
        if memory_cleanup_task:
            memory_cleanup_task.cancel()
            try:
                await asyncio.wait_for(memory_cleanup_task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        
        if settings.cleanup_on_shutdown:
            logger.info("Performing cleanup on shutdown")
            try:
                result = force_cleanup()
                logger.info(f"Shutdown cleanup complete: {result}")
            except Exception as e:
                logger.error(f"Error during shutdown cleanup: {e}")
        
        logger.info("Shutdown complete")

    app = FastAPI(
        title="Classifarr Image Embedding Service",
        version=__version__,
        lifespan=lifespan,
    )

    app.state.embedder = embedder_instance
    app.state.queue = queue
    app.state.settings = settings
    app.state.logger = logger

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger = get_logger("image_embedder.errors")
        logger.exception(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
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
        embedder_instance: ImageEmbedder = app.state.embedder
        queue: EmbedQueue = app.state.queue
        
        device_info = embedder_instance.get_device_info()
        model_status = embedder_instance.get_model_status()
        memory_info = embedder_instance.get_memory_info()
        
        return HealthResponse(
            status="ok",
            provider="local",
            default_model=settings.default_model,
            device=DeviceInfo(**device_info) if device_info else None,
            models=[ModelStatus(**m) for m in model_status],
            memory=MemoryInfo(**memory_info) if memory_info else None,
            queue=queue.stats().__dict__,
        )

    @app.get("/ready", response_model=ReadyResponse)
    def ready() -> ReadyResponse:
        embedder_instance: ImageEmbedder = app.state.embedder
        device_info = embedder_instance.get_device_info()
        default_loaded = embedder_instance.is_default_model_loaded()
        
        return ReadyResponse(
            ready=default_loaded,
            default_model_loaded=default_loaded,
            device=DeviceInfo(**device_info) if device_info else None,
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

    @app.post("/admin/cleanup", response_model=CleanupResponse)
    async def trigger_cleanup():
        logger = app.state.logger
        logger.info("Manual cleanup triggered")
        result = force_cleanup()
        mem_usage = get_memory_usage()
        return CleanupResponse(
            gc_collected=result["gc_collected"],
            gpu_freed_mb=result["gpu_freed_mb"],
            process_rss_mb=mem_usage["process_rss_mb"],
            gpu_allocated_mb=mem_usage["gpu_allocated_mb"],
            gpu_reserved_mb=mem_usage["gpu_reserved_mb"],
        )

    @app.post("/embed-image", response_model=EmbedImageResponse)
    async def embed_image(payload: EmbedImageRequest, response: Response):
        logger = app.state.logger
        embedder_instance: ImageEmbedder = app.state.embedder
        queue: EmbedQueue = app.state.queue
        
        if not payload.image_url and not payload.image_base64:
            raise HTTPException(
                status_code=400,
                detail="image_url or image_base64 is required"
            )
        try:
            await queue.acquire()
            await queue.acquire_shared()
            try:
                embedding, dims, provider, model_name, image_size = await anyio.to_thread.run_sync(  # type: ignore[union-attr]
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
        except ValueError as exc:
            logger.warning(f"Validation error: {exc}")
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception(f"Embedding error: {exc}")
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
