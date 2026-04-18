# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Admin endpoints: POST /admin/cleanup."""

from fastapi import APIRouter, Depends, Request

from ..memory import force_cleanup, get_memory_usage
from ..models import CleanupResponse


def make_router(auth) -> APIRouter:
    router = APIRouter()

    @router.post("/admin/cleanup", response_model=CleanupResponse, dependencies=[Depends(auth)])
    async def trigger_cleanup(request: Request):
        logger = request.app.state.logger
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

    return router
