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

"""Contract tests: run the real tools and assert the parsers still match.

The synthetic parser tests in test_parsers.py validate parsing logic against
recorded output; these tests validate the recording itself. Each test runs an
actual tool on a tiny fixture with a known violation and asserts that at least
one structured diagnostic comes out. When a tool release changes its output
format, these tests fail instead of the toolchain silently degrading to raw
output dumps.

Excluded tools: pyright (multi-second node startup blows the per-test budget),
deptry (needs a full project context), pip-audit (needs network access), and
bun (optional dependency). Their parsers are covered synthetically, and the
runtime parser-drift warning covers format changes at execution time.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from toolchain.checker import merge_output
from toolchain.checkers import _parse_mdformat_file_list
from toolchain.parsers import (
    parse_bandit,
    parse_biome,
    parse_biome_files,
    parse_mdformat,
    parse_pytest,
    parse_ruff_format,
    parse_ruff_format_files,
    parse_ruff_json,
    parse_ty,
    parse_vulture,
)

_REPO_ROOT = Path(__file__).parents[2]
_BIOME_BIN = _REPO_ROOT / "node_modules" / ".bin" / "biome"

_RUFF = shutil.which("ruff")
_TY = shutil.which("ty")
_MDFORMAT = shutil.which("mdformat")
_VULTURE = shutil.which("vulture")
_BANDIT = shutil.which("bandit")


def _run(command: list[str], cwd: Path | None = None) -> tuple[str, int]:
    """Run a tool, returning (merged output, exit code)."""
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=8,
        cwd=cwd,
        check=False,
    )
    return merge_output(result.stdout, result.stderr), result.returncode


@pytest.mark.skipif(_RUFF is None, reason="ruff not on PATH")
def test_ruff_check_json_contract(tmp_path: Path) -> None:
    """`ruff check --output-format=json` parses into located diagnostics."""
    assert _RUFF is not None
    bad = tmp_path / "bad.py"
    bad.write_text("import os\n", encoding="utf-8")

    output, code = _run(
        [
            _RUFF,
            "check",
            "--isolated",
            "--select",
            "F401",
            "--output-format=json",
            str(bad),
        ]
    )
    assert code == 1

    diagnostics = parse_ruff_json(output, code)
    located = [d for d in diagnostics if d.location is not None]
    assert located, f"parser extracted nothing from: {output[:500]}"
    assert located[0].location is not None
    assert located[0].location.file.endswith("bad.py")
    assert located[0].location.line == 1
    assert "[F401]" in located[0].message


@pytest.mark.skipif(_RUFF is None, reason="ruff not on PATH")
def test_ruff_format_check_contract(tmp_path: Path) -> None:
    """`ruff format --check` still reports `Would reformat:` lines."""
    assert _RUFF is not None
    bad = tmp_path / "bad.py"
    bad.write_text("x=1\n", encoding="utf-8")

    output, code = _run([_RUFF, "format", "--check", "--isolated", str(bad)])
    assert code == 1

    files = parse_ruff_format_files(output)
    assert files == [str(bad)], (
        f"file list parser extracted nothing from: {output[:500]}"
    )
    diagnostics = parse_ruff_format(output, code)
    assert diagnostics


@pytest.mark.skipif(_TY is None, reason="ty not on PATH")
def test_ty_contract(tmp_path: Path) -> None:
    """ty failure output parses into located diagnostics."""
    assert _TY is not None
    bad = tmp_path / "bad.py"
    bad.write_text('x: int = "s"\n', encoding="utf-8")

    output, code = _run([_TY, "check", str(bad)], cwd=tmp_path)
    assert code != 0

    diagnostics = parse_ty(output, code)
    assert diagnostics, f"parser extracted nothing from: {output[:500]}"
    assert diagnostics[0].location is not None
    assert diagnostics[0].location.file.endswith("bad.py")
    assert diagnostics[0].location.line == 1


def test_pytest_contract(tmp_path: Path) -> None:
    """pytest failure output parses into per-test diagnostics."""
    test_file = tmp_path / "test_contract_sample.py"
    test_file.write_text(
        "def test_fails():\n    assert 1 == 2\n",
        encoding="utf-8",
    )

    # Plugin autoload off: only core pytest output matters for the contract
    # and subprocess startup stays well within the test timeout.
    env = {**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(test_file.name),
            "-o",
            "addopts=",
            "--tb=short",
            "--no-header",
            "-q",
        ],
        capture_output=True,
        text=True,
        timeout=8,
        cwd=tmp_path,
        env=env,
        check=False,
    )
    assert result.returncode == 1

    diagnostics = parse_pytest(
        merge_output(result.stdout, result.stderr), result.returncode
    )
    assert diagnostics, f"parser extracted nothing from: {result.stdout[:500]}"
    assert any("test_fails" in d.message for d in diagnostics)


@pytest.mark.skipif(_MDFORMAT is None, reason="mdformat not on PATH")
def test_mdformat_check_contract(tmp_path: Path) -> None:
    """`mdformat --check` failure output yields the unformatted file list.

    The fixture path is deliberately long: mdformat word-wraps its error
    message, and the wrap points shift with path length, so a long path
    exercises the wrapping behavior on every environment.
    """
    assert _MDFORMAT is not None
    nested = tmp_path / "a-rather-long-directory-name-to-force-message-wrapping"
    nested.mkdir()
    bad = nested / "bad.md"
    bad.write_text("# Title\n* bullet\n", encoding="utf-8")

    output, code = _run([_MDFORMAT, "--check", str(bad)])
    assert code != 0

    files = _parse_mdformat_file_list(output)
    assert files == [str(bad)], (
        f"file list parser extracted nothing from: {output[:500]}"
    )
    diagnostics = parse_mdformat(output, code)
    assert diagnostics


@pytest.mark.skipif(_VULTURE is None, reason="vulture not on PATH")
def test_vulture_contract(tmp_path: Path) -> None:
    """vulture output parses into located diagnostics."""
    assert _VULTURE is not None
    bad = tmp_path / "dead.py"
    bad.write_text("def unused_function():\n    return 1\n", encoding="utf-8")

    output, code = _run([_VULTURE, str(bad), "--min-confidence", "0"])
    assert code != 0

    diagnostics = parse_vulture(output, code)
    assert diagnostics, f"parser extracted nothing from: {output[:500]}"
    assert any("unused_function" in d.message for d in diagnostics)
    assert diagnostics[0].location is not None
    assert diagnostics[0].location.file.endswith("dead.py")


@pytest.mark.skipif(_BANDIT is None, reason="bandit not on PATH")
def test_bandit_contract(tmp_path: Path) -> None:
    """bandit text output parses into located diagnostics."""
    assert _BANDIT is not None
    bad = tmp_path / "insecure.py"
    bad.write_text(
        "import subprocess\nsubprocess.call('ls', shell=True)\n",
        encoding="utf-8",
    )

    output, code = _run([_BANDIT, str(bad)])
    assert code == 1

    diagnostics = parse_bandit(output, code)
    assert diagnostics, f"parser extracted nothing from: {output[:500]}"
    assert diagnostics[0].location is not None
    assert diagnostics[0].location.file.endswith("insecure.py")


@pytest.mark.skipif(
    not _BIOME_BIN.exists(), reason="biome not installed (run npm install)"
)
def test_biome_github_reporter_contract(tmp_path: Path) -> None:
    """`biome check --reporter=github` parses into located diagnostics."""
    # Mirror the repo layout so the biome.json file includes match.
    static = tmp_path / "src" / "weakincentives" / "cli" / "static"
    static.mkdir(parents=True)
    bad = static / "bad.js"
    bad.write_text("var x = 1;\nif (x == 1) { console.log(x) }\n", encoding="utf-8")

    output, code = _run(
        [
            str(_BIOME_BIN),
            "check",
            "--reporter=github",
            f"--config-path={_REPO_ROOT}",
            str(static),
        ],
        cwd=tmp_path,
    )
    assert code != 0

    diagnostics = parse_biome(output, code)
    assert diagnostics, f"parser extracted nothing from: {output[:500]}"
    assert diagnostics[0].location is not None
    assert diagnostics[0].location.line is not None
    assert parse_biome_files(output), "file list parser extracted nothing"
