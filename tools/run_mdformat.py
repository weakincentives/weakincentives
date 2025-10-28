"""Run mdformat in check mode over all Markdown files."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _collect_markdown_files(root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "*.md"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"git ls-files failed: {message}")

    files = [root / line for line in result.stdout.splitlines() if line]
    return sorted(files)


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    markdown_files = _collect_markdown_files(project_root)

    if not markdown_files:
        return 0

    command = ["mdformat", "--check", *map(str, markdown_files)]
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
