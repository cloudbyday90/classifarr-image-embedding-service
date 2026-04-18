# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""App lifespan: signal handling, model warmup, and background memory cleanup."""

import asyncio
import signal
from contextlib import asynccontextmanager

import anyio
from fastapi import FastAPI

from .memory import cleanup_gpu_memory, force_cleanup, check_memory_health


def make_lifespan(embedder_instance, settings, logger, batch_window=None):
    """Return a FastAPI lifespan context manager bound to the given embedder, settings, and logger."""

    shutdown_event = asyncio.Event()

    async def _memory_cleanup_loop():
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
        memory_cleanup_task: asyncio.Task | None = None
        loop = asyncio.get_event_loop()

        def _signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown")
            shutdown_event.set()

        try:
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(sig, lambda s=sig: _signal_handler(s, None))
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

        if batch_window is not None:
            await batch_window.start()

        memory_cleanup_task = asyncio.create_task(_memory_cleanup_loop())
        logger.info("Started memory cleanup background task")

        yield

        logger.info("Shutdown initiated")
        shutdown_event.set()

        if batch_window is not None:
            await batch_window.stop()

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

    return lifespan
