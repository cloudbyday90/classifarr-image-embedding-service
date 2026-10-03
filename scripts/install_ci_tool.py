# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Install reviewed CI tools as a non-root user; CodeQL stays a verified archive."""

import argparse
import os
import platform
import shutil
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory

from ci_downloads import verified_download
from ci_tool_contracts import tool_contract


def extract_executable(archive: Path, name: str, destination: Path) -> None:
    with tarfile.open(archive, "r:gz") as tar:
        members = [member for member in tar if member.name == name]
        if (
            len(members) != 1
            or not members[0].isfile()
            or not 0 < members[0].size <= 512 * 1024**2
        ):
            raise ValueError("Archive must contain one bounded regular executable")
        # Never unpack archive paths, symlinks, device nodes or ancillary files.
        source = tar.extractfile(members[0])
        if source is None:
            raise ValueError("Archive executable could not be read")
        with source, destination.open("xb") as output:
            shutil.copyfileobj(source, output, length=1024 * 1024)


def install(name: str, directory: Path) -> Path:
    if platform.system() != "Linux" or platform.machine().lower() not in {
        "x86_64",
        "amd64",
    }:
        raise ValueError("CI tools support Linux amd64 only")
    contract = tool_contract(name)
    directory.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".verified-", dir=directory) as temporary:
        archive = Path(temporary) / "download"
        verified_download(
            contract["url"], archive, contract["sha256"], contract["max_bytes"]
        )
        staged = Path(temporary) / contract["executable"]
        if contract["kind"] == "tar-member":
            extract_executable(archive, contract["executable"], staged)
        else:
            archive.rename(staged)
        staged.chmod(0o600 if contract["kind"] == "archive" else 0o700)
        target = directory / contract["executable"]
        os.replace(staged, target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tool", choices=("gitleaks", "trivy", "buildx", "codeql"))
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    target = install(args.tool, args.directory.resolve())
    print(f"Verified {args.tool}: {target}")


if __name__ == "__main__":
    main()
