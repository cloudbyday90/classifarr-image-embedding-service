# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

from copy import deepcopy

import pytest
from dependency_artifacts import WheelArtifact
from dependency_audit_report import verify_audit_report
from test_dependency_artifacts import PIP


def clean_report():
    return {
        "dependencies": [{"name": "pip", "version": PIP["version"], "vulns": []}],
        "fixes": [],
    }


def test_exact_audit_inventory_passes():
    verify_audit_report(clean_report(), [WheelArtifact(**PIP)])


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d.update(dependencies=[]),
        lambda d: d["dependencies"].append(deepcopy(d["dependencies"][0])),
        lambda d: d["dependencies"][0].update(skip_reason="not found"),
        lambda d: d["dependencies"][0].update(vulns=[{"id": "PYSEC-fixture"}]),
        lambda d: d["dependencies"][0].update(version="26.1"),
        lambda d: d["dependencies"][0].update(name="unexpected"),
        lambda d: d["dependencies"][0].pop("vulns"),
    ],
)
def test_success_exit_cannot_hide_incomplete_or_different_audits(mutation):
    report = clean_report()
    mutation(report)
    with pytest.raises(ValueError):
        verify_audit_report(report, [WheelArtifact(**PIP)])
