# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Serialize process-shared lazy imports/export setup without locking warm inference."""

import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import wraps
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
T = TypeVar("T")
_initialization_lock = threading.RLock()


@contextmanager
def initialization_guard() -> Iterator[None]:
    # Library lazy modules/conversion state are shared by all owners in a process.
    # Reentrancy allows the complete model load to call individually guarded loaders.
    with _initialization_lock:
        yield


def serialized_initialization(function: Callable[P, T]) -> Callable[P, T]:
    @wraps(function)
    def guarded(*args: P.args, **kwargs: P.kwargs) -> T:
        with initialization_guard():
            return function(*args, **kwargs)

    return guarded
