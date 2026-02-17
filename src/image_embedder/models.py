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
    image_size: Optional[int] = Field(default=None, gt=0, description="Resize shortest edge before embed")


class EmbedImageResponse(BaseModel):
    embedding: List[float]
    dims: int
    provider: str
    model: str
    image_size: int


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
