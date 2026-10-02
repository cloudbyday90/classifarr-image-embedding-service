# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Shared container entrypoint with explicit worker and connection budgets."""

import uvicorn

from .config import Settings


def main() -> None:
    settings = Settings()
    uvicorn.run(
        "image_embedder.main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.server_workers,
        limit_concurrency=settings.server_concurrency,
        backlog=settings.server_backlog,
    )


if __name__ == "__main__":
    main()
