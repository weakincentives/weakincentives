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

"""Generic contract suite for Shell protocol implementations.

Mirrors ``tests.helpers.filesystem.FilesystemValidationSuite`` for the
shell facet: any :class:`~weakincentives.sandbox.Shell` implementation
must satisfy these argv, cwd, env, stdin, timeout, output-cap, and
launch-failure semantics. Subclasses provide a :class:`ShellHarness`
binding the implementation to a host directory the tests can inspect::

    from tests.helpers.shell import ShellHarness, ShellValidationSuite

    class TestMyShell(ShellValidationSuite):
        @pytest.fixture
        def harness(self, tmp_path: Path) -> ShellHarness:
            return ShellHarness(root=tmp_path, make=...)

Implementation-specific behavior (host env hygiene, clock injection)
stays in the implementation's own test module.
"""

from __future__ import annotations

import sys
from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pytest

from weakincentives.sandbox import Shell


class ShellFactory(Protocol):
    """Builds a Shell rooted at the harness directory."""

    def __call__(
        self,
        *,
        env: Mapping[str, str] | None = None,
        default_timeout_s: float | None = None,
        max_output_bytes: int | None = None,
    ) -> Shell:
        """Build a shell; ``None`` keeps the implementation default."""
        ...


@dataclass(frozen=True)
class ShellHarness:
    """A Shell factory bound to the host directory commands run in."""

    root: Path
    make: ShellFactory


def python_argv(*code: str) -> list[str]:
    """Build an argv vector running inline Python code."""
    return [sys.executable, "-c", "\n".join(code)]


class ShellValidationSuite:
    """Contract tests every Shell implementation must pass.

    Subclasses implement the ``harness`` fixture. The directory behind
    ``harness.root`` must be the (initially empty) root commands run
    under, reachable from the test for setup and assertions.
    """

    @pytest.fixture
    @abstractmethod
    def harness(self, tmp_path: Path) -> ShellHarness:
        """Provide the shell factory and its root directory."""
        ...

    # --- Basic execution ---------------------------------------------------

    def test_captures_stdout_and_exit_code(self, harness: ShellHarness) -> None:
        result = harness.make().run(python_argv("print('hello')"))
        assert result.ok
        assert result.exit_code == 0
        assert result.stdout == b"hello\n"
        assert result.stderr == b""
        assert not result.truncated
        assert not result.timed_out

    def test_captures_stderr_and_nonzero_exit(self, harness: ShellHarness) -> None:
        result = harness.make().run(
            python_argv("import sys", "sys.stderr.write('boom')", "sys.exit(3)")
        )
        assert result.exit_code == 3
        assert result.stderr == b"boom"
        assert not result.ok

    def test_no_shell_interpretation(self, harness: ShellHarness) -> None:
        result = harness.make().run([sys.executable, "-c", "print('$HOME *')"])
        assert result.stdout == b"$HOME *\n"

    def test_runs_at_root_by_default(self, harness: ShellHarness) -> None:
        result = harness.make().run(python_argv("import os", "print(os.getcwd())"))
        assert result.stdout.decode().strip() == str(harness.root.resolve())

    def test_stdin_piped(self, harness: ShellHarness) -> None:
        result = harness.make().run(
            python_argv("import sys", "sys.stdout.write(sys.stdin.read())"),
            stdin=b"piped input",
        )
        assert result.stdout == b"piped input"

    def test_duration_non_negative(self, harness: ShellHarness) -> None:
        result = harness.make().run(python_argv("pass"))
        assert result.duration_s >= 0.0

    def test_empty_argv_rejected(self, harness: ShellHarness) -> None:
        with pytest.raises(ValueError, match="argv must not be empty"):
            harness.make().run([])

    # --- Working directory ---------------------------------------------------

    def test_relative_cwd_inside_root(self, harness: ShellHarness) -> None:
        (harness.root / "sub").mkdir()
        result = harness.make().run(
            python_argv("import os", "print(os.getcwd())"), cwd="sub"
        )
        expected = (harness.root / "sub").resolve()
        assert result.stdout.decode().strip() == str(expected)

    def test_absolute_cwd_rejected(self, harness: ShellHarness) -> None:
        with pytest.raises(ValueError, match="must be relative"):
            harness.make().run(python_argv("pass"), cwd=str(harness.root))

    def test_escaping_cwd_rejected(self, harness: ShellHarness) -> None:
        with pytest.raises(PermissionError, match="escapes the sandbox root"):
            harness.make().run(python_argv("pass"), cwd="../outside")

    def test_missing_cwd_rejected(self, harness: ShellHarness) -> None:
        with pytest.raises(FileNotFoundError, match="does not exist"):
            harness.make().run(python_argv("pass"), cwd="nope")

    def test_dot_cwd_is_root(self, harness: ShellHarness) -> None:
        result = harness.make().run(
            python_argv("import os", "print(os.getcwd())"), cwd="."
        )
        assert result.stdout.decode().strip() == str(harness.root.resolve())

    # --- Environment ---------------------------------------------------------

    def test_constructor_env_applied(self, harness: ShellHarness) -> None:
        shell = harness.make(env={"WINK_SHELL_VAR": "ctor"})
        result = shell.run(
            python_argv("import os", "print(os.environ['WINK_SHELL_VAR'])")
        )
        assert result.stdout.decode().strip() == "ctor"

    def test_run_env_overlays_constructor_env(self, harness: ShellHarness) -> None:
        shell = harness.make(env={"WINK_SHELL_VAR": "ctor"})
        result = shell.run(
            python_argv("import os", "print(os.environ['WINK_SHELL_VAR'])"),
            env={"WINK_SHELL_VAR": "run"},
        )
        assert result.stdout.decode().strip() == "run"

    # --- Timeouts -------------------------------------------------------------

    def test_timeout_kills_and_reports(self, harness: ShellHarness) -> None:
        result = harness.make().run(
            python_argv("import time", "time.sleep(30)"), timeout_s=0.2
        )
        assert result.timed_out
        assert result.exit_code == 124
        assert not result.ok

    def test_timeout_preserves_partial_output(self, harness: ShellHarness) -> None:
        result = harness.make().run(
            python_argv(
                "import sys, time",
                "print('partial', flush=True)",
                "time.sleep(30)",
            ),
            timeout_s=1.0,
        )
        assert result.timed_out
        assert b"partial" in result.stdout

    def test_default_timeout_applies_when_none(self, harness: ShellHarness) -> None:
        shell = harness.make(default_timeout_s=0.2)
        result = shell.run(python_argv("import time", "time.sleep(30)"))
        assert result.timed_out

    def test_non_positive_timeout_rejected(self, harness: ShellHarness) -> None:
        with pytest.raises(ValueError, match="timeout_s must be positive"):
            harness.make().run(python_argv("pass"), timeout_s=0)

    def test_non_positive_default_timeout_rejected(self, harness: ShellHarness) -> None:
        with pytest.raises(ValueError, match="default_timeout_s must be positive"):
            harness.make(default_timeout_s=0)

    # --- Output caps -----------------------------------------------------------

    def test_stdout_capped_and_flagged(self, harness: ShellHarness) -> None:
        shell = harness.make(max_output_bytes=16)
        result = shell.run(python_argv("print('x' * 100)"))
        assert result.truncated
        assert len(result.stdout) == 16

    def test_stderr_capped_and_flagged(self, harness: ShellHarness) -> None:
        shell = harness.make(max_output_bytes=16)
        result = shell.run(python_argv("import sys", "sys.stderr.write('e' * 100)"))
        assert result.truncated
        assert len(result.stderr) == 16

    def test_output_under_cap_not_flagged(self, harness: ShellHarness) -> None:
        shell = harness.make(max_output_bytes=16)
        result = shell.run(python_argv("print('ok')"))
        assert not result.truncated

    def test_non_positive_cap_rejected(self, harness: ShellHarness) -> None:
        with pytest.raises(ValueError, match="max_output_bytes must be positive"):
            harness.make(max_output_bytes=0)

    # --- Launch failures ---------------------------------------------------------

    def test_missing_executable_exits_127(self, harness: ShellHarness) -> None:
        result = harness.make().run(["wink-definitely-not-a-command"])
        assert result.exit_code == 127
        assert b"command not found" in result.stderr

    def test_non_executable_file_exits_126(self, harness: ShellHarness) -> None:
        script = harness.root / "script.sh"
        script.write_text("#!/bin/sh\necho hi\n")
        script.chmod(0o644)
        result = harness.make().run([str(script)])
        assert result.exit_code == 126
        assert b"permission denied" in result.stderr


__all__ = [
    "ShellFactory",
    "ShellHarness",
    "ShellValidationSuite",
    "python_argv",
]
