# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Require a complete, unskipped, vulnerability-free pip-audit inventory."""

from dependency_artifacts import WheelArtifact, canonical_name


def verify_audit_report(report: dict, artifacts: list[WheelArtifact]) -> None:
    records = report.get("dependencies") if isinstance(report, dict) else None
    if not isinstance(records, list) or len(records) != len(artifacts):
        raise ValueError("Audit report has an incomplete dependency inventory")
    actual = {}
    for record in records:
        if (
            not isinstance(record, dict)
            or record.get("skip_reason")
            or record.get("vulns") != []
        ):
            raise ValueError(
                "Audit report contains skipped, invalid or vulnerable packages"
            )
        raw_name = record.get("name")
        if not isinstance(raw_name, str):
            raise ValueError("Audit report has an invalid distribution name")
        name = canonical_name(raw_name)
        if name in actual or not isinstance(record.get("version"), str):
            raise ValueError("Audit report has duplicate or invalid package identities")
        actual[name] = record["version"]
    if actual != {item.name: item.version for item in artifacts}:
        raise ValueError("Audit inventory differs from the attested graph")
