#!/usr/bin/env python3
# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Update the committed coverage baseline to the current coverage numbers.

Run this after intentionally improving or accepting a coverage change:

    python scripts/update_coverage_baseline.py
    git add .coverage-baseline.json
    git commit -m "chore: update coverage baseline"
"""

import json
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
COVERAGE_XML = ROOT / "coverage.xml"
BASELINE_PATH = ROOT / ".coverage-baseline.json"


def parse_coverage_xml(path: Path) -> dict:
    root = ET.parse(path).getroot()
    return {
        "lines": float(root.attrib["line-rate"]) * 100,
        "branches": float(root.attrib.get("branch-rate", 0)) * 100,
    }


def main() -> None:
    if not COVERAGE_XML.exists():
        print(f"coverage.xml not found at {COVERAGE_XML}. Run pytest first.")
        sys.exit(1)

    coverage = parse_coverage_xml(COVERAGE_XML)
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **coverage,
    }

    BASELINE_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Coverage baseline updated: {BASELINE_PATH.relative_to(ROOT)}")
    for metric, value in coverage.items():
        print(f"  {metric}: {value:.2f}%")


if __name__ == "__main__":
    main()
