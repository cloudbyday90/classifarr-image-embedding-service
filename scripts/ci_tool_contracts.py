# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Load strict native tool contracts without accepting ambient versions or caches."""

import json
import re
from pathlib import Path

from ci_downloads import validate_asset_url

ROOT = Path(__file__).resolve().parents[1]


def tool_contract(name: str, manifest: Path = ROOT / ".github/ci-tools.json") -> dict:
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or data.get("platform") != "linux-amd64":
        raise ValueError("Unknown CI tool manifest contract")
    tool = data["tools"][name]
    validate_asset_url(tool["url"])
    if (
        not re.fullmatch(r"[a-f0-9]{64}", tool["sha256"])
        or tool["kind"] not in {"binary", "archive", "tar-member"}
        or not re.fullmatch(r"[A-Za-z0-9_.-]+", tool["executable"])
        or type(tool["max_bytes"]) is not int
        or not 0 < tool["max_bytes"] <= 3 * 1024**3
    ):
        raise ValueError("Invalid CI tool artifact contract")
    parsed_path = tool["url"].split("?", 1)[0]
    if (
        not parsed_path.startswith("https://github.com/")
        or "/releases/download/" not in parsed_path
    ):
        raise ValueError("Initial download must name an official release asset")
    return tool
