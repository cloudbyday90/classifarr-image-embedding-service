# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Bounded HTTPS downloads from reviewed GitHub release asset destinations."""

import hashlib
import time
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

ALLOWED_HOSTS = {
    "github.com",
    "release-assets.githubusercontent.com",
    "objects.githubusercontent.com",
}


def validate_asset_url(url: str) -> None:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in ALLOWED_HOSTS
        or parsed.username
        or parsed.password
        or parsed.port not in (None, 443)
        or parsed.fragment
        or any(ord(c) < 32 or c == "\\" for c in url)
    ):
        raise ValueError("Unapproved release asset URL")


class AssetRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validate_asset_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def verified_download(url: str, destination: Path, digest: str, max_bytes: int) -> None:
    validate_asset_url(url)
    opener = build_opener(ProxyHandler({}), AssetRedirects())
    request = Request(
        url,
        headers={"User-Agent": "classifarr-ci-verifier", "Accept-Encoding": "identity"},
    )
    deadline = time.monotonic() + 600
    size = 0
    checksum = hashlib.sha256()
    # The caller owns a private temporary directory. Nothing executable is published yet.
    with opener.open(request, timeout=30) as response, destination.open("xb") as output:
        if response.status != 200:
            raise ValueError("Release asset download did not return 200")
        validate_asset_url(response.url)
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("Release asset download exceeded ten minutes")
            chunk = response.read1(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > max_bytes:
                raise ValueError("Release asset exceeds its reviewed size ceiling")
            checksum.update(chunk)
            output.write(chunk)
    if not size or checksum.hexdigest() != digest:
        raise ValueError("Release asset checksum differs from the reviewed contract")
