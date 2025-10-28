from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _run_ty_check(tmp_path: Path, source: str) -> subprocess.CompletedProcess[str]:
    module = tmp_path / "snippet.py"
    module.write_text(source)
    return subprocess.run(
        [sys.executable, "-m", "ty", "check", str(module)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )


def test_section_rejects_non_dataclass_type_parameter(tmp_path: Path) -> None:
    result = _run_ty_check(
        tmp_path,
        "from weakincentives.prompts.section import Section\n\nSection[str]\n",
    )

    assert result.returncode != 0
    combined_output = result.stdout + result.stderr
    assert "Section" in combined_output
    assert "str" in combined_output


def test_tool_rejects_non_dataclass_type_parameters(tmp_path: Path) -> None:
    result = _run_ty_check(
        tmp_path,
        """
from dataclasses import dataclass

from weakincentives.prompts.tool import Tool

@dataclass
class ExampleResult:
    value: str

Tool[str, ExampleResult]
""",
    )

    assert result.returncode != 0
    combined_output = result.stdout + result.stderr
    assert "Tool" in combined_output
    assert "str" in combined_output
