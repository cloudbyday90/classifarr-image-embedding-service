# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

from typing import List, Optional
from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    id: str
    name: str
    dims: int
    image_size: int


class EmbedImageRequest(BaseModel):
    image_url: Optional[str] = Field(default=None, description="Remote image URL")
    image_base64: Optional[str] = Field(default=None, description="Base64-encoded image bytes")
    model: Optional[str] = Field(default=None, description="Model name, e.g., ViT-L-14")
    normalize: bool = Field(default=True, description="L2 normalize embeddings")
    image_size: Optional[int] = Field(default=None, description="Resize shortest edge before embed")


class EmbedImageResponse(BaseModel):
    embedding: List[float]
    dims: int
    provider: str
    model: str
    image_size: int


class HealthResponse(BaseModel):
    status: str
    provider: str
    default_model: str
