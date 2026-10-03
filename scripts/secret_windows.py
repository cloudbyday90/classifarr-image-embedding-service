# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Create Windows files with a protected current-user DACL before writing data."""

import ctypes
import os
from ctypes import wintypes
from pathlib import Path
from typing import Any


class _SecurityAttributes(ctypes.Structure):
    _fields_ = [
        ("length", wintypes.DWORD),
        ("descriptor", ctypes.c_void_p),
        ("inherit_handle", wintypes.BOOL),
    ]


class _SidAndAttributes(ctypes.Structure):
    _fields_ = [("sid", ctypes.c_void_p), ("attributes", wintypes.DWORD)]


def _bind(library: Any, name: str, result: Any, arguments: list[Any]) -> Any:
    function = getattr(library, name)
    function.restype = result
    function.argtypes = arguments
    return function


def _current_user_sid(kernel: Any, security: Any) -> str:
    open_token = _bind(
        security,
        "OpenProcessToken",
        wintypes.BOOL,
        [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)],
    )
    get_token = _bind(
        security,
        "GetTokenInformation",
        wintypes.BOOL,
        [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
        ],
    )
    convert_sid = _bind(
        security,
        "ConvertSidToStringSidW",
        wintypes.BOOL,
        [ctypes.c_void_p, ctypes.POINTER(wintypes.LPWSTR)],
    )
    token = wintypes.HANDLE()
    if not open_token(wintypes.HANDLE(-1), 0x0008, ctypes.byref(token)):  # TOKEN_QUERY
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        needed = wintypes.DWORD()
        get_token(token, 1, None, 0, ctypes.byref(needed))  # TokenUser
        if ctypes.get_last_error() != 122 or not (
            ctypes.sizeof(_SidAndAttributes) <= needed.value <= 65536
        ):
            raise OSError("Cannot size current-user token information")
        buffer = ctypes.create_string_buffer(needed.value)
        if not get_token(token, 1, buffer, needed.value, ctypes.byref(needed)):
            raise ctypes.WinError(ctypes.get_last_error())
        user = ctypes.cast(buffer, ctypes.POINTER(_SidAndAttributes)).contents
        sid = wintypes.LPWSTR()
        if not convert_sid(user.sid, ctypes.byref(sid)):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            if sid.value is None:
                raise OSError("Current-user SID is unavailable")
            return sid.value
        finally:
            kernel.LocalFree(ctypes.cast(sid, ctypes.c_void_p))
    finally:
        kernel.CloseHandle(token)


def open_private_windows(path: Path) -> int:
    """Return a noninheritable write fd; unsupported ACL filesystems fail closed."""
    if os.name != "nt":
        raise OSError("Windows secret creation requires Windows")
    import msvcrt

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    security = ctypes.WinDLL("advapi32", use_last_error=True)
    _bind(kernel, "CloseHandle", wintypes.BOOL, [wintypes.HANDLE])
    _bind(kernel, "LocalFree", ctypes.c_void_p, [ctypes.c_void_p])
    convert = _bind(
        security,
        "ConvertStringSecurityDescriptorToSecurityDescriptorW",
        wintypes.BOOL,
        [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(wintypes.DWORD),
        ],
    )
    create = _bind(
        kernel,
        "CreateFileW",
        wintypes.HANDLE,
        [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.POINTER(_SecurityAttributes),
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        ],
    )
    volume_info = _bind(
        kernel,
        "GetVolumeInformationByHandleW",
        wintypes.BOOL,
        [
            wintypes.HANDLE,
            wintypes.LPWSTR,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
            ctypes.POINTER(wintypes.DWORD),
            ctypes.POINTER(wintypes.DWORD),
            wintypes.LPWSTR,
            wintypes.DWORD,
        ],
    )
    sid = _current_user_sid(kernel, security)
    descriptor = ctypes.c_void_p()
    # P prevents inheritance of broader parent ACEs. No SYSTEM/admin ACE is added.
    if not convert(f"O:{sid}D:P(A;;FA;;;{sid})", 1, ctypes.byref(descriptor), None):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        attributes = _SecurityAttributes(
            ctypes.sizeof(_SecurityAttributes), descriptor, False
        )
        handle = create(
            str(path), 0x40000000, 0, ctypes.byref(attributes), 1, 0x80, None
        )
        if handle == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        fd = None
        try:
            flags = wintypes.DWORD()
            if not volume_info(
                handle, None, 0, None, None, ctypes.byref(flags), None, 0
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            if not flags.value & 0x00000008:  # FILE_PERSISTENT_ACLS
                raise OSError("Secret setup requires a persistent-ACL filesystem")
            fd = msvcrt.open_osfhandle(handle, os.O_WRONLY | os.O_BINARY)
            handle = None  # CRT now owns the native handle.
            os.set_inheritable(fd, False)
            return fd
        except BaseException:
            if fd is not None:
                os.close(fd)
            elif handle is not None:
                kernel.CloseHandle(handle)
            path.unlink(missing_ok=True)  # Only the file successfully created above.
            raise
    finally:
        kernel.LocalFree(descriptor)
