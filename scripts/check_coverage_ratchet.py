#!/usr/bin/env python3
# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Check that test coverage has not regressed below the committed baseline.

Usage (CI / local check):
    python scripts/check_coverage_ratchet.py

If this fails due to an intentional coverage change, update the baseline:
    python scripts/update_coverage_baseline.py
    git add .coverage-baseline.json
    git commit -m "chore: update coverage baseline"
"""

import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).parent.parent
COVERAGE_XML = ROOT / "coverage.xml"
BASELINE_PATH = ROOT / ".coverage-baseline.json"

# Treat deltas within 0.05 percentage points as equal to avoid CI flakiness
# (coverage tools can vary slightly between environments).
EPSILON = 0.05
METRICS = ["lines", "branches"]


def parse_coverage_xml(path: Path) -> dict:
    root = ET.parse(path).getroot()
    return {
        "lines": float(root.attrib["line-rate"]) * 100,
        "branches": float(root.attrib.get("branch-rate", 0)) * 100,
    }


def write_step_summary(rows: list, regressions: list) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    lines = [
        "### Coverage Ratchet",
        "",
        "| Metric | Baseline | Current | Delta | Status |",
        "|---|---:|---:|---:|---|",
    ]
    for row in rows:
        delta_str = f"{'+' if row['delta'] >= 0 else ''}{row['delta']:.2f}"
        status = "❌ regressed" if row["delta"] < -EPSILON else "✅ ok"
        lines.append(
            f"| {row['metric']} | {row['baseline']:.2f}% | {row['current']:.2f}% | {delta_str} | {status} |"
        )
    lines.append("")
    if regressions:
        lines.append("**Result: ❌ failed**")
        for issue in regressions:
            lines.append(f"- {issue}")
    else:
        lines.append("**Result: ✅ passed**")
    lines.append("")
    with open(summary_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    if not COVERAGE_XML.exists():
        print(f"coverage.xml not found at {COVERAGE_XML}. Run pytest first.")
        sys.exit(1)

    if not BASELINE_PATH.exists():
        print(f"No baseline found at {BASELINE_PATH}.")
        print("Run: python scripts/update_coverage_baseline.py")
        sys.exit(1)

    current = parse_coverage_xml(COVERAGE_XML)
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    is_ci = os.environ.get("GITHUB_ACTIONS") == "true"
    baseline_rel = BASELINE_PATH.relative_to(ROOT)

    regressions = []
    rows = []

    for metric in METRICS:
        prev = baseline.get(metric)
        now = current.get(metric, 0.0)
        if prev is None:
            regressions.append(f"Missing baseline value for '{metric}' in {baseline_rel}")
            continue
        delta = now - prev
        rows.append({"metric": metric, "baseline": prev, "current": now, "delta": delta})
        if delta < -EPSILON:
            msg = f"{metric} regressed: baseline={prev:.2f}% current={now:.2f}%"
            regressions.append(msg)
            if is_ci:
                print(f"::error file={baseline_rel},title=Coverage Ratchet::{msg}")

    print("Coverage ratchet summary:")
    for row in rows:
        delta_str = f"{'+' if row['delta'] >= 0 else ''}{row['delta']:.2f}"
        print(f"  {row['metric']:<10} baseline={row['baseline']:.2f}%  current={row['current']:.2f}%  delta={delta_str}")

    write_step_summary(rows, regressions)

    if regressions:
        print("\nCoverage ratchet FAILED:")
        for issue in regressions:
            print(f"  - {issue}")
        print("\nIf this reduction is intentional, run:")
        print("  python scripts/update_coverage_baseline.py")
        print("  git add .coverage-baseline.json && git commit -m 'chore: update coverage baseline'")
        sys.exit(1)

    print("\nCoverage ratchet passed — no regressions detected.")


if __name__ == "__main__":
    main()
