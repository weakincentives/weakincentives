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

"""Tests for the LocalShell facet and CommandResult.

The generic Shell contract lives in
``tests.helpers.shell.ShellValidationSuite`` and runs here over
``LocalShell``; only host-specific behavior (env hygiene and capture
semantics, clock injection) is tested directly.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

from tests.helpers.shell import ShellHarness, ShellValidationSuite, python_argv
from weakincentives.clock import FakeClock
from weakincentives.sandbox import (
    DEFAULT_SHELL_TIMEOUT_S,
    MAX_COMMAND_OUTPUT_BYTES,
    CommandResult,
    LocalShell,
    Shell,
)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path


class TestShellProtocol:
    def test_local_shell_implements_shell(self, root: Path) -> None:
        assert isinstance(LocalShell(root), Shell)

    def test_defaults_exported(self) -> None:
        assert DEFAULT_SHELL_TIMEOUT_S == 60.0
        assert MAX_COMMAND_OUTPUT_BYTES == 1024 * 1024


class TestCommandResult:
    def test_ok_for_zero_exit(self) -> None:
        result = CommandResult(
            exit_code=0, stdout=b"", stderr=b"", truncated=False, duration_s=0.1
        )
        assert result.ok

    def test_not_ok_for_nonzero_exit(self) -> None:
        result = CommandResult(
            exit_code=1, stdout=b"", stderr=b"", truncated=False, duration_s=0.1
        )
        assert not result.ok

    def test_not_ok_when_timed_out(self) -> None:
        result = CommandResult(
            exit_code=0,
            stdout=b"",
            stderr=b"",
            truncated=False,
            duration_s=0.1,
            timed_out=True,
        )
        assert not result.ok


class TestLocalShellContract(ShellValidationSuite):
    """The generic Shell contract over LocalShell."""

    @pytest.fixture
    def harness(self, tmp_path: Path) -> ShellHarness:
        def make(
            *,
            env: Mapping[str, str] | None = None,
            default_timeout_s: float | None = None,
            max_output_bytes: int | None = None,
        ) -> Shell:
            return LocalShell(
                tmp_path,
                env=env,
                default_timeout_s=(
                    default_timeout_s
                    if default_timeout_s is not None
                    else DEFAULT_SHELL_TIMEOUT_S
                ),
                max_output_bytes=(
                    max_output_bytes
                    if max_output_bytes is not None
                    else MAX_COMMAND_OUTPUT_BYTES
                ),
            )

        return ShellHarness(root=tmp_path, make=make)


class TestLocalShellSpecific:
    """Host-environment behavior beyond the generic contract."""

    def test_git_variables_stripped(
        self, root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GIT_DIR", "/tmp/should-not-leak")
        monkeypatch.setenv("WINK_SHELL_KEEP", "kept")
        shell = LocalShell(root)
        result = shell.run(
            python_argv(
                "import os",
                "print('GIT_DIR' in os.environ, os.environ.get('WINK_SHELL_KEEP'))",
            )
        )
        assert result.stdout.decode().strip() == "False kept"

    def test_environment_captured_at_construction(
        self, root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        shell = LocalShell(root)
        monkeypatch.setenv("WINK_SHELL_LATE", "late")
        result = shell.run(
            python_argv("import os", "print(os.environ.get('WINK_SHELL_LATE'))")
        )
        assert result.stdout.decode().strip() == "None"

    def test_duration_measured_with_injected_clock(self, root: Path) -> None:
        result = LocalShell(root, clock=FakeClock()).run(python_argv("pass"))
        assert result.duration_s == 0.0

    def test_root_property_resolved(self, root: Path) -> None:
        assert LocalShell(root).root == str(root.resolve())
