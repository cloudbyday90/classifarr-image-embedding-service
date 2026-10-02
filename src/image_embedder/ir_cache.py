# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Versioned, verified publication of complete OpenVINO XML/BIN pairs."""

import hashlib
import json
import math
import os
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory

from filelock import FileLock

from .artifact_integrity import sha256_file

IR_FILES = ("model.xml", "model.bin")
MAX_MANIFEST_BYTES = 64 * 1024


def contract_key(contract: dict) -> str:
    encoded = json.dumps(
        contract, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _file_record(path: Path) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
        raise ValueError("IR artifacts must be nonempty regular files")
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _matches(entry: Path, contract: dict) -> bool:
    try:
        manifest_path = entry / "manifest.json"
        if (
            entry.is_symlink()
            or manifest_path.is_symlink()
            or not manifest_path.is_file()
        ):
            return False
        with manifest_path.open("rb") as stream:
            data = stream.read(MAX_MANIFEST_BYTES + 1)
        if len(data) > MAX_MANIFEST_BYTES:
            return False
        manifest = json.loads(data)
        return manifest == {
            "contract": contract,
            "files": {name: _file_record(entry / name) for name in IR_FILES},
        }
    except (OSError, ValueError, TypeError, RecursionError):
        return False


class IRCache:
    def __init__(self, root: Path, *, lock_timeout: float = 300) -> None:
        if not math.isfinite(lock_timeout) or lock_timeout < 0:
            raise ValueError("IR lock timeout must be finite and nonnegative")
        self.root = root
        self.lock_timeout = lock_timeout

    def get_or_create(self, contract: dict, export: Callable[[Path], None]) -> Path:
        key = contract_key(contract)
        self.root.mkdir(parents=True, exist_ok=True)
        entry = self.root / key
        # Separate persistent lock inode; never unlink a lock used by waiters.
        with FileLock(str(self.root / f"{key}.lock"), timeout=self.lock_timeout):
            if _matches(entry, contract):
                return entry / "model.xml"
            with TemporaryDirectory(
                prefix=f".build-{key[:12]}-", dir=self.root
            ) as temp:
                staging = Path(temp) / "complete"
                staging.mkdir()
                export(staging / "model.xml")
                manifest = {
                    "contract": contract,
                    "files": {name: _file_record(staging / name) for name in IR_FILES},
                }
                (staging / "manifest.json").write_text(
                    json.dumps(manifest, sort_keys=True, allow_nan=False),
                    encoding="utf-8",
                )
                # Only replace after the new pair and manifest are complete.
                if entry.exists() or entry.is_symlink():
                    os.replace(entry, Path(temp) / "invalid")
                os.replace(staging, entry)
            return entry / "model.xml"
