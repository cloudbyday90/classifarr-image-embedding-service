# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Audit every attested Windows graph without resolving it on a Linux interpreter."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from dependency_audit_report import verify_audit_report
from dependency_locks import load_lock
from dependency_profiles import PROFILES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output_directory.mkdir(parents=True, exist_ok=True)
    for profile in PROFILES.values():
        if profile.backend != "windows-contracts":
            continue
        _, artifacts = load_lock(root, profile)
        requirements = args.output_directory / f"{profile.name}.txt"
        report = args.output_directory / f"{profile.name}.json"
        # This is an audit inventory, never an installation or cross-target resolution.
        requirements.write_bytes(
            "".join(
                f"{item.name}=={item.version} --hash=sha256:{item.sha256}\n"
                for item in artifacts
            ).encode()
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip_audit",
                "--strict",
                "--disable-pip",
                "--require-hashes",
                "--vulnerability-service",
                "osv",
                "--progress-spinner",
                "off",
                "--format",
                "json",
                "--output",
                str(report),
                "--cache-dir",
                str(args.output_directory / "cache"),
                "-r",
                str(requirements),
            ],
            check=True,
        )
        if report.stat().st_size > 1024 * 1024:
            raise ValueError("Audit report is oversized")
        verify_audit_report(json.loads(report.read_text(encoding="utf-8")), artifacts)
        print(f"Audited complete {profile.name}: {len(artifacts)} packages")


if __name__ == "__main__":
    main()
