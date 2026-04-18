# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Model catalogue endpoint: GET /models."""

from fastapi import APIRouter, Depends, Request

from ..embedder import ImageEmbedder
from ..models import ModelInfo


def make_router(auth) -> APIRouter:
    router = APIRouter()

    @router.get("/models", response_model=list[ModelInfo], dependencies=[Depends(auth)])
    def list_models(request: Request):
        embedder_instance: ImageEmbedder = request.app.state.embedder
        return [
            ModelInfo(
                id=spec.name,
                name=spec.name,
                dims=spec.dims,
                image_size=spec.image_size,
            )
            for spec in embedder_instance.list_models()
        ]

    return router
