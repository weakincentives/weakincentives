"""Run mdformat over repository Markdown files while keeping success output quiet."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable

EXCLUDED_DIRECTORIES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
}


def _iter_markdown_files(root: Path) -> Iterable[Path]:
    for directory_path, dirnames, filenames in os.walk(root):
        current = Path(directory_path)
        dirnames[:] = [
            name
            for name in dirnames
            if name not in EXCLUDED_DIRECTORIES and not name.startswith("__pycache__")
        ]

        for filename in filenames:
            if filename.endswith(".md"):
                yield current / filename


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    markdown_files = sorted(_iter_markdown_files(project_root))

    if not markdown_files:
        return 0

    command = [
        sys.executable,
        "-m",
        "mdformat",
        "--check",
        *[str(path) for path in markdown_files],
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    output = f"{result.stdout}{result.stderr}".strip()

    if result.returncode != 0 and output:
        print(output)

    return result.returncode


if __name__ == "__main__":  # pragma: no cover - invoked via Makefile target
    raise SystemExit(main())
