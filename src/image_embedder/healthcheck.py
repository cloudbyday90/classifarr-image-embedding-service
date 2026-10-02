# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Probe the configured local listener without loading inference libraries."""

from urllib.request import ProxyHandler, build_opener

from .config import Settings


def main() -> None:
    settings = Settings()
    host = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(settings.host, settings.host)
    if ":" in host:
        host = f"[{host}]"
    opener = build_opener(ProxyHandler({}))
    with opener.open(f"http://{host}:{settings.port}/health", timeout=5):
        pass


if __name__ == "__main__":
    main()
