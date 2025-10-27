"""Run deptry on the package while keeping successful runs quiet."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    package_path = project_root / "src" / "weakincentives"

    command = [
        sys.executable,
        "-m",
        "deptry",
        "--no-ansi",
        "--per-rule-ignores",
        "DEP002=openai",
        str(package_path),
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    output = f"{result.stdout}{result.stderr}".strip()

    if (result.returncode != 0 or "warning" in output.lower()) and output:
        print(output)

    return result.returncode


if __name__ == "__main__":  # pragma: no cover - invoked via Makefile target
    raise SystemExit(main())
