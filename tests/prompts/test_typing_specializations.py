from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from textwrap import dedent


def _run_ty(snippet: str, directory: Path) -> subprocess.CompletedProcess[str]:
    target = directory / "snippet.py"
    target.write_text(dedent(snippet))
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "ty",
            "check",
            "--python",
            sys.executable,
            str(target),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_section_rejects_non_dataclass_specialization(tmp_path: Path) -> None:
    result = _run_ty(
        """
        from weakincentives.prompts import Section

        Section[str]
        """,
        tmp_path,
    )

    assert result.returncode != 0
    assert "Expected `SupportsDataclass`, found `str`" in result.stdout


def test_tool_requires_dataclass_parameters(tmp_path: Path) -> None:
    result = _run_ty(
        """
        from weakincentives.prompts import Tool

        Tool[str, str]
        """,
        tmp_path,
    )

    assert result.returncode != 0
    assert "Expected `SupportsDataclass`, found `str`" in result.stdout


def test_tool_accepts_dataclass_parameters(tmp_path: Path) -> None:
    result = _run_ty(
        """
        from dataclasses import dataclass

        from weakincentives.prompts import Tool


        @dataclass
        class Params:
            value: str


        @dataclass
        class Result:
            message: str


        Tool[Params, Result]
        """,
        tmp_path,
    )

    assert result.returncode == 0
    assert "All checks passed!" in result.stdout
