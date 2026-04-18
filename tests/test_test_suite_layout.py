# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Guardrails for the repository's pytest import conventions."""

import ast
from pathlib import Path


TESTS_DIR = Path(__file__).resolve().parent


def test_test_modules_do_not_import_test_helpers_via_tests_package():
    """Keep helper imports aligned with pytest.ini's `pythonpath = src tests`.

    Under the current layout, helper modules inside ``tests/`` are importable as
    top-level modules such as ``fakes``. Importing them as ``tests.fakes``
    depends on the repository root being on ``sys.path``, which CI does not
    guarantee when invoking plain ``pytest``.
    """

    offenders: list[str] = []

    for path in sorted(TESTS_DIR.glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module == "tests" or node.module.startswith("tests."):
                    offenders.append(f"{path.name}:{node.lineno} imports from {node.module!r}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "tests" or alias.name.startswith("tests."):
                        offenders.append(f"{path.name}:{node.lineno} imports {alias.name!r}")

    assert not offenders, (
        "Test modules must import shared helpers via the configured top-level "
        "test path (for example `from fakes import ...`), not via `tests.*`:\n"
        + "\n".join(offenders)
    )
