# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Resolve with reviewed pip inside each target interpreter; emit complete wheel locks."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from dependency_artifacts import report_artifacts
from dependency_locks import load_lock, write_lock
from dependency_profiles import PROFILES, check_environment, selected_profile


def resolve(
    arguments: list[str], report: Path, root: Path, cache: Path | None = None
) -> dict:
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("PIP_")
    }
    environment["PIP_CONFIG_FILE"] = os.devnull
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "--isolated",
            "--disable-pip-version-check",
            *(["--cache-dir", str(cache)] if cache else []),
            "install",
            "--no-input",
            "--dry-run",
            "--ignore-installed",
            "--only-binary=:all:",
            "--report",
            str(report),
            *arguments,
        ],
        cwd=root,
        env=environment,
        check=True,
    )
    return json.loads(report.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("bootstrap", "cpu", "qa", "cuda", "cuda-legacy", "openvino", "audit"),
    )
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Optional writable pip cache for native generation",
    )
    parser.add_argument(
        "--constraints",
        type=Path,
        help="Reviewed installed versions; omit deliberately for a full refresh",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check all committed locks without resolving/downloading",
    )
    args = parser.parse_args()
    if args.check:
        for profile in PROFILES.values():
            _, artifacts = load_lock(args.root, profile)
            print(f"Valid {profile.name}: {len(artifacts)} wheels")
        return
    if not args.backend:
        parser.error("Provide --backend for target-native generation")
    profile = selected_profile(args.backend)
    check_environment(profile)
    root = args.root.resolve()
    with TemporaryDirectory(prefix="classifarr-lock-report-") as temporary:
        report = Path(temporary) / "environment.json"
        arguments = [
            "--index-url",
            "https://pypi.org/simple",
            "-r",
            "requirements-bootstrap.txt",
        ]
        for name in profile.inputs:
            if name != "requirements-bootstrap.txt":
                arguments.extend(["-r", name])
        if args.constraints:
            arguments.extend(["-c", str(args.constraints.resolve())])
        if profile.torch_version:
            assert profile.torch_index is not None
            torch_input = next(
                name
                for name in profile.inputs
                if name.startswith("requirements-torch-")
            )
            torch_report = resolve(
                ["--no-deps", "--index-url", profile.torch_index, "-r", torch_input],
                Path(temporary) / "torch.json",
                root,
                args.cache_dir,
            )
            _, selected = report_artifacts(torch_report, profile)
            if len(selected) != 1:
                raise ValueError("Torch seed must contain exactly one vendor wheel")
            torch = selected[0]
            arguments.append(f"torch @ {torch.url}#sha256={torch.sha256}")
        environment, artifacts = report_artifacts(
            resolve(arguments, report, root, args.cache_dir), profile
        )
        write_lock(root, profile, environment, artifacts)
    print(f"Generated {profile.name}: {len(artifacts)} wheels")


if __name__ == "__main__":
    main()
