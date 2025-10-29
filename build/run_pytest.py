"""Pytest runner that emits output only for errors and warnings."""

from __future__ import annotations

import sys
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO

import pytest


def main() -> int:
    args = sys.argv[1:]
    buffer = StringIO()

    with redirect_stdout(buffer), redirect_stderr(buffer):
        code = pytest.main(args)

    output = buffer.getvalue()
    if code != 0 or "warning" in output.lower():
        print(output, end="")

    return code


if __name__ == "__main__":  # pragma: no cover - exercised via Makefile target
    sys.exit(main())
