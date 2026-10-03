# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Install or verify a reviewed binary-only environment without index resolution."""

import argparse
import os
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

from dependency_locks import load_lock, verify_inventory
from dependency_profiles import check_environment, selected_profile


def pip_command(*arguments: str, target: Path | None = None) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "pip",
        "--isolated",
        "--disable-pip-version-check",
        "install",
        "--no-input",
        "--no-index",
        "--require-hashes",
        "--only-binary=:all:",
    ]
    if target:
        command.extend(["--target", str(target)])
    return [*command, *arguments]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        required=True,
        choices=("bootstrap", "cpu", "qa", "cuda", "cuda-legacy", "openvino", "audit"),
    )
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--target", type=Path, help="Empty isolated directory; used for audit tooling"
    )
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    profile = selected_profile(args.backend)
    check_environment(profile)
    lock_file, artifacts = load_lock(args.root, profile)
    if (
        args.target
        and not args.verify_only
        and args.target.exists()
        and any(args.target.iterdir())
    ):
        parser.error("The installation target must be empty")
    if not args.verify_only:
        environment = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("PIP_")
        }
        environment["PIP_CONFIG_FILE"] = os.devnull
        subprocess.run(
            pip_command(
                *(["--force-reinstall"] if profile.backend == "bootstrap" else []),
                "-r",
                str(lock_file),
                target=args.target,
            ),
            env=environment,
            check=True,
        )
    # The bootstrap may coexist with the venv seed's setuptools; full profiles may not.
    if profile.backend == "bootstrap":
        if args.target:
            verify_inventory(artifacts, args.target)
        elif version("pip") != "26.2.1":
            raise ValueError(
                "Bootstrap installer version differs from reviewed pip 26.2.1"
            )
    else:
        if args.target:
            verify_inventory(artifacts, args.target)
            environment = dict(os.environ, PYTHONPATH=str(args.target))
            subprocess.run(
                [sys.executable, "-m", "pip", "--isolated", "check"],
                env=environment,
                check=True,
            )
        else:
            subprocess.run(
                [sys.executable, "-m", "pip", "--isolated", "check"], check=True
            )
            verify_inventory(artifacts)
    print(f"Verified dependency profile {profile.name}: {len(artifacts)} hashed wheels")


if __name__ == "__main__":
    main()
