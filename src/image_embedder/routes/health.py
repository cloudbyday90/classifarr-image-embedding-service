# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Health and readiness endpoints: GET /health, GET /ready."""

from fastapi import APIRouter, Request

from ..embedder import ImageEmbedder
from ..models import DeviceInfo, HealthResponse, MemoryInfo, ModelStatus, ReadyResponse
from ..queue import EmbedQueue


def make_router(limiter, rate_limit_health: str) -> APIRouter:
    router = APIRouter()

    @router.get("/health", response_model=HealthResponse)
    @limiter.limit(rate_limit_health)
    def health(request: Request) -> HealthResponse:
        embedder_instance: ImageEmbedder = request.app.state.embedder
        queue: EmbedQueue = request.app.state.queue
        settings = request.app.state.settings

        device_info = embedder_instance.get_device_info()
        model_status = embedder_instance.get_model_status()
        memory_info = embedder_instance.get_memory_info()
        cache_info = embedder_instance.get_cache_info()

        return HealthResponse(
            status="ok",
            provider="local",
            default_model=settings.default_model,
            device=DeviceInfo(**device_info) if device_info else None,
            models=[ModelStatus(**m) for m in model_status],
            memory=MemoryInfo(**memory_info) if memory_info else None,
            queue=queue.stats().__dict__,
            cache=cache_info,
        )

    @router.get("/ready", response_model=ReadyResponse)
    @limiter.limit(rate_limit_health)
    def ready(request: Request) -> ReadyResponse:
        embedder_instance: ImageEmbedder = request.app.state.embedder
        device_info = embedder_instance.get_device_info()
        default_loaded = embedder_instance.is_default_model_loaded()

        return ReadyResponse(
            ready=default_loaded,
            default_model_loaded=default_loaded,
            device=DeviceInfo(**device_info) if device_info else None,
        )

    return router
