# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Validate the complete tag inventory before selecting any cleanup deletions."""

import re
from datetime import datetime
from urllib.parse import parse_qsl, urlsplit

NAMESPACE = "cloudbyday90"
REPOSITORY = "classifarr-image-embedder"
LIST_PATH = f"/v2/namespaces/{NAMESPACE}/repositories/{REPOSITORY}/tags"
DELETE_PATH = f"/v2/repositories/{NAMESPACE}/{REPOSITORY}/tags"


def page_path(url: str) -> str:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.netloc != "hub.docker.com"
        or parsed.path.rstrip("/") != LIST_PATH
        or parsed.fragment
        or any(ord(c) < 32 or c == "\\" for c in url)
    ):
        raise ValueError("Docker Hub pagination left the approved tag inventory")
    query = parse_qsl(parsed.query, strict_parsing=True)
    if len({k for k, _ in query}) != len(query) or any(
        key not in {"page", "page_size"}
        or not value.isascii()
        or not value.isdecimal()
        or not 1 <= int(value) <= (100 if key == "page_size" else 100)
        for key, value in query
    ):
        raise ValueError("Invalid Docker Hub pagination parameters")
    return parsed.path + ("?" + parsed.query if parsed.query else "")


def inventory(client) -> list[dict]:
    path = LIST_PATH + "?page_size=100"
    visited = set()
    tags = []
    total = None
    while path:
        if path in visited or len(visited) >= 100:
            raise ValueError("Docker Hub pagination loop or page limit")
        visited.add(path)
        result = client.request("GET", path)
        count = result.get("count")
        if (
            type(count) is not int
            or not 0 <= count <= 10000
            or (total is not None and count != total)
        ):
            raise ValueError("Docker Hub inventory count is invalid or changed")
        total = count
        records = result.get("results")
        if not isinstance(records, list) or len(records) > 100:
            raise ValueError("Docker Hub tag page is invalid")
        tags.extend(records)
        next_url = result.get("next")
        if next_url is not None and (not isinstance(next_url, str) or not next_url):
            raise ValueError("Docker Hub next page is invalid")
        path = page_path(next_url) if next_url else None
    if len(tags) != total:
        raise ValueError("Docker Hub inventory is incomplete")
    return tags


def deletion_plan(tags: list[dict], current_tag: str, keep: int = 5) -> list[str]:
    if keep < 1 or not re.fullmatch(r"[\w][\w.-]{0,127}", current_tag, re.ASCII):
        raise ValueError("Invalid Docker Hub retention policy")
    seen = set()
    dated = []
    for tag in tags:
        if not isinstance(tag, dict):
            raise ValueError("Invalid Docker Hub tag record")
        name, date = tag.get("name"), tag.get("last_updated")
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"[\w][\w.-]{0,127}", name, re.ASCII)
            or name in seen
        ):
            raise ValueError("Invalid or duplicate Docker Hub tag name")
        seen.add(name)
        if not isinstance(date, str):
            raise ValueError("Docker Hub tag timestamp is missing")
        stamp = datetime.fromisoformat(date.replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            raise ValueError("Docker Hub tag timestamp has no timezone")
        if name != "latest":
            dated.append((stamp, name))
    if current_tag not in seen:
        raise ValueError("Current release tag is absent; cleanup refused")
    ordered = [name for _, name in sorted(dated, reverse=True)]
    protected = set(ordered[:keep]) | {"latest", current_tag}
    return [name for name in reversed(ordered) if name not in protected]
