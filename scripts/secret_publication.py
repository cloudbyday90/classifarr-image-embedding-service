# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Private same-directory staging and complete-file, exclusive publication."""

import os
import secrets
import stat
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator


class SetupError(Exception):
    """A setup policy failure with a safe, nonsecret diagnostic."""


def _is_reparse(info: os.stat_result) -> bool:
    return bool(getattr(info, "st_file_attributes", 0) & 0x400)


def validate_root(root: Path) -> None:
    info = root.lstat()
    if not stat.S_ISDIR(info.st_mode) or _is_reparse(info):
        raise SetupError("Setup directory must be a regular directory, not a link.")
    if os.name == "posix":
        if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o022:
            raise SetupError(
                "Setup directory must be owned by you and not writable by group/others."
            )
    elif os.name != "nt":
        raise SetupError(
            "Secret setup supports POSIX and Windows ACL filesystems only."
        )


def validate_target(path: Path) -> bool:
    """Return presence without following links or reading an existing secret."""
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(info.st_mode) or _is_reparse(info) or info.st_nlink != 1:
        raise SetupError(
            "Setup targets must be regular files without links or reparse points."
        )
    if os.name == "posix" and info.st_uid != os.geteuid():
        raise SetupError("Setup targets must be owned by you.")
    return True


@contextmanager
def setup_lock(root: Path) -> Iterator[None]:
    """Serialize cooperating creators/rotators; never remove another setup's lock."""
    lock = root / ".env.setup.lock"
    try:
        lock.mkdir(mode=0o700)
    except FileExistsError:
        raise SetupError(
            "Setup is running or .env.setup.lock remains from an interrupted run. "
            "Inspect it and remove only the empty lock after confirming no setup is running."
        ) from None
    try:
        yield
    finally:
        lock.rmdir()


def _open_private(path: Path) -> int:
    if os.name == "nt":
        from secret_windows import open_private_windows

        return open_private_windows(path)
    if os.name != "posix":
        raise SetupError("Unsupported secret file platform.")
    fd = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600
    )
    try:
        os.fchmod(
            fd, 0o600
        )  # Exact owner permissions even under an unusually strict umask.
        info = os.fstat(fd)
        if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o600:
            raise SetupError(
                "Secret file owner-only permissions could not be established."
            )
        return fd
    except BaseException:
        os.close(fd)
        path.unlink(missing_ok=True)
        raise


def publish_private(path: Path, content: bytes, *, replace: bool = False) -> None:
    """Publish complete bytes; an existing file wins unless rotation is explicit."""
    _publish(path, content, _open_private, replace=replace)


def _open_defaults(path: Path) -> int:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if os.name == "posix":
        flags |= os.O_NOFOLLOW | os.O_CLOEXEC
    fd = os.open(path, flags, 0o644)
    try:
        if os.name == "posix":
            os.fchmod(
                fd, 0o644
            )  # Nonsecret Docker config must be readable by its user.
        return fd
    except BaseException:
        os.close(fd)
        path.unlink(missing_ok=True)
        raise


def publish_defaults(path: Path, content: bytes) -> None:
    """Publish nonsecret defaults without replacement; keep container read access."""
    _publish(path, content, _open_defaults, replace=False)


def _publish(
    path: Path, content: bytes, opener: Callable[[Path], int], *, replace: bool
) -> None:
    temporary = path.parent / f".env.setup-{secrets.token_hex(16)}.tmp"
    fd = opener(temporary)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            validate_target(path)
            os.replace(temporary, path)
        else:
            os.link(temporary, path, follow_symlinks=False)
    finally:
        temporary.unlink(missing_ok=True)
