from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET_DIRS = ["src", "tests"]
HEADER_LINES = [
    "# Classifarr Image Embedding Service - companion service for Classifarr",
    "# Copyright (C) 2024-2026 Classifarr Contributors",
    "# SPDX-License-Identifier: GPL-3.0-or-later",
]


def has_header(text: str) -> bool:
    lines = text.splitlines()
    if len(lines) < len(HEADER_LINES):
        return False
    return lines[: len(HEADER_LINES)] == HEADER_LINES


def main() -> int:
    missing = []
    for directory in TARGET_DIRS:
        path = ROOT / directory
        if not path.exists():
            continue
        for file_path in path.rglob("*.py"):
            if "__pycache__" in file_path.parts:
                continue
            content = file_path.read_text(encoding="utf-8")
            if not has_header(content):
                missing.append(file_path.relative_to(ROOT))

    if missing:
        print("Missing or incorrect copyright headers:")
        for item in missing:
            print(f"- {item}")
        return 1

    print("All checked files include the expected copyright header.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
