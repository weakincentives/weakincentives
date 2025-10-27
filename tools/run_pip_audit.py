"""Run pip-audit with the repository defaults and quiet successes."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent

    command = [
        sys.executable,
        "-m",
        "pip_audit",
        "--progress-spinner",
        "off",
        "--strict",
        "--skip-editable",
        str(project_root),
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    output = f"{result.stdout}{result.stderr}".strip()

    if (result.returncode != 0 or "warning" in output.lower()) and output:
        print(output)

    return result.returncode


if __name__ == "__main__":  # pragma: no cover - invoked via Makefile target
    raise SystemExit(main())
