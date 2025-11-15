# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
        from weakincentives.prompt import Section

        Section[str]
        """,
        tmp_path,
    )

    assert result.returncode != 0
    assert "Expected `SupportsDataclass`, found `str`" in result.stdout


def test_tool_requires_dataclass_parameters(tmp_path: Path) -> None:
    result = _run_ty(
        """
        from weakincentives.prompt import Tool

        Tool[str, str]
        """,
        tmp_path,
    )

    assert result.returncode != 0
    assert "Expected `SupportsDataclass`, found `str`" in result.stdout


def test_tool_requires_dataclass_results(tmp_path: Path) -> None:
    result = _run_ty(
        """
        from dataclasses import dataclass

        from weakincentives.prompt import Tool


        @dataclass
        class Params:
            value: str


        Tool[Params, str]
        """,
        tmp_path,
    )

    assert result.returncode != 0
    assert (
        "Expected `SupportsDataclass | Sequence[SupportsDataclass]`, found `str`"
        in result.stdout
    )


def test_tool_accepts_dataclass_parameters(tmp_path: Path) -> None:
    result = _run_ty(
        """
        from dataclasses import dataclass

        from weakincentives.prompt import Tool


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
