# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Reject missing, malformed and incomplete OSV reports before comparison."""

import json
import os
from pathlib import Path


def check_report(path: Path, outcome: str) -> None:
    if outcome not in {"success", "failure"}:
        raise ValueError("OSV scan did not run to completion")
    if not 0 < path.stat().st_size <= 32 * 1024**2:
        raise ValueError("OSV report is empty or exceeds its limit")
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or not isinstance(report.get("results"), list):
        raise ValueError("OSV report has no valid results inventory")


def main() -> None:
    if os.environ.get("DIFFERENTIAL") == "true":
        check_report(Path("old-results.json"), os.environ.get("OLD_OUTCOME", ""))
    check_report(Path("new-results.json"), os.environ.get("NEW_OUTCOME", ""))


if __name__ == "__main__":
    main()
