# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Run the fixed native Windows suite with complete inventory and outcome gates."""

import argparse
import json
import os
import platform
import sys
from importlib.metadata import distributions
from pathlib import Path

from dependency_locks import load_lock, verify_inventory
from dependency_profiles import check_environment, selected_profile
from windows_contract_policy import SUITES, WindowsContractPolicy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    profile = selected_profile("windows-contracts")
    check_environment(profile)
    _, artifacts = load_lock(root, profile)
    verify_inventory(artifacts)
    os.chdir(root)
    import pytest

    policy = WindowsContractPolicy()
    result = int(
        pytest.main(
            [
                *SUITES,
                "-o",
                "addopts=",
                "-p",
                "no:cacheprovider",
                "-o",
                "faulthandler_timeout=45",
            ],
            plugins=[policy],
        )
    )
    report = {
        "schema_version": 1,
        "profile": profile.name,
        "python": sys.version,
        "platform": platform.platform(),
        "runner_image": {
            key: os.environ.get(key) for key in ("ImageOS", "ImageVersion")
        },
        "packages": sorted(
            (
                {"name": dist.metadata["Name"], "version": dist.version}
                for dist in distributions()
            ),
            key=lambda item: item["name"].lower(),
        ),
        "exit_code": result,
        "gate_errors": policy.errors,
        "outcomes": policy.outcomes,
        "deselected_uvloop": policy.deselected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(
        (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    )
    return result


if __name__ == "__main__":
    raise SystemExit(main())
