# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

from typing import List, Optional, Self
from pydantic import BaseModel, Field, model_validator


class ModelInfo(BaseModel):
    id: str
    name: str
    dims: int
    image_size: int


class EmbedImageRequest(BaseModel):
    image_url: Optional[str] = Field(default=None, description="Remote image URL")
    image_base64: Optional[str] = Field(default=None, description="Base64-encoded image bytes")
    model: Optional[str] = Field(
        default=None,
        pattern=r"^[a-zA-Z0-9][a-zA-Z0-9\-_\.]*$",
        description="Model name, e.g., ViT-L-14",
    )
    normalize: bool = Field(default=True, description="L2 normalize embeddings")
    image_size: Optional[int] = Field(default=None, gt=0, description="Resize shortest edge before embed")

    @model_validator(mode="after")
    def validate_single_image_source(self) -> Self:
        if (self.image_url is None) == (self.image_base64 is None):
            raise ValueError("Exactly one of image_url or image_base64 is required")
        return self


class EmbedImageResponse(BaseModel):
    embedding: List[float]
    dims: int
    provider: str
    model: str
    image_size: int


class EmbedBatchItem(BaseModel):
    image_url: Optional[str] = Field(default=None, description="Remote image URL")
    image_base64: Optional[str] = Field(default=None, description="Base64-encoded image bytes")

    @model_validator(mode="after")
    def validate_single_image_source(self) -> Self:
        if (self.image_url is None) == (self.image_base64 is None):
            raise ValueError("Exactly one of image_url or image_base64 is required")
        return self


class EmbedBatchRequest(BaseModel):
    items: List[EmbedBatchItem] = Field(..., min_length=1, description="Images to embed (up to the configured limit)")
    model: Optional[str] = Field(
        default=None,
        pattern=r"^[a-zA-Z0-9][a-zA-Z0-9\-_\.]*$",
        description="Model to use for all items, e.g., ViT-L-14",
    )
    normalize: bool = Field(default=True, description="L2 normalize all embeddings")
    image_size: Optional[int] = Field(default=None, gt=0, description="Resize shortest edge for all items")


class EmbedBatchItemResult(BaseModel):
    index: int = Field(description="Zero-based position in the request items array")
    status: str = Field(description='"ok" or "error"')
    embedding: Optional[List[float]] = None
    dims: Optional[int] = None
    model: Optional[str] = None
    image_size: Optional[int] = None
    error: Optional[str] = Field(default=None, description="Error message when status is 'error'")


class EmbedBatchResponse(BaseModel):
    model: str
    image_size: int
    total: int
    succeeded: int
    failed: int
    results: List[EmbedBatchItemResult]


class ModelStatus(BaseModel):
    name: str
    loaded: bool


class DeviceInfo(BaseModel):
    type: str
    name: Optional[str] = None


class MemoryInfo(BaseModel):
    allocated_mb: Optional[float] = None
    reserved_mb: Optional[float] = None


class HealthResponse(BaseModel):
    status: str
    provider: str
    default_model: str
    device: Optional[DeviceInfo] = None
    models: List[ModelStatus] = Field(default_factory=list)
    memory: Optional[MemoryInfo] = None
    queue: dict = Field(default_factory=dict, description="Queue status and concurrency hints")
    cache: Optional[dict] = Field(default=None, description="Embedding cache statistics; null when caching is disabled")


class ReadyResponse(BaseModel):
    ready: bool
    default_model_loaded: bool
    device: Optional[DeviceInfo] = None


class CleanupResponse(BaseModel):
    gc_collected: int
    gpu_freed_mb: float
    process_rss_mb: Optional[float] = None
    gpu_allocated_mb: Optional[float] = None
    gpu_reserved_mb: Optional[float] = None
