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

"""Shell facet: the sandbox's command-execution surface.

``Shell`` is the narrow exec primitive of the environment: it runs an
``argv`` vector — the name denotes the surface, not ``/bin/sh``. No shell
interpretation (globbing, quoting, ``$VAR`` expansion, pipes) ever happens;
remote shells in later milestones implement the same contract over a
transport.

``LocalShell`` runs subprocesses rooted at a host directory with output
caps and a default timeout. Failures that a POSIX shell reports through
exit codes keep that shape here: a missing executable yields exit code
127, a non-executable target 126, and a timeout 124 (with
``timed_out=True``), always returning a :class:`CommandResult` rather than
raising.
"""

from __future__ import annotations

import os
import subprocess  # nosec: B404
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol, runtime_checkable

from ..clock import SYSTEM_CLOCK, MonotonicClock

__all__ = [
    "DEFAULT_SHELL_TIMEOUT_S",
    "MAX_COMMAND_OUTPUT_BYTES",
    "CommandResult",
    "LocalShell",
    "Shell",
]

DEFAULT_SHELL_TIMEOUT_S: Final[float] = 60.0
"""Timeout applied when ``run`` is called with ``timeout_s=None``."""

MAX_COMMAND_OUTPUT_BYTES: Final[int] = 1024 * 1024
"""Default per-stream cap on captured stdout/stderr bytes."""

_EXIT_TIMEOUT: Final[int] = 124
_EXIT_NOT_EXECUTABLE: Final[int] = 126
_EXIT_NOT_FOUND: Final[int] = 127


@dataclass(slots=True, frozen=True)
class CommandResult:
    """Outcome of one :meth:`Shell.run` invocation.

    Attributes:
        exit_code: Process exit code. Shell-conventional codes are used for
            launch failures: 124 timeout, 126 not executable, 127 not found.
        stdout: Captured standard output, capped at the shell's output limit.
        stderr: Captured standard error, capped at the shell's output limit.
        truncated: True when either stream exceeded the cap and was cut.
        duration_s: Wall time spent running the command.
        timed_out: True when the command was killed by the timeout.
    """

    exit_code: int
    stdout: bytes
    stderr: bytes
    truncated: bool
    duration_s: float
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        """True when the command exited zero without timing out."""
        return self.exit_code == 0 and not self.timed_out


@runtime_checkable
class Shell(Protocol):
    """Command-execution facet vended by a sandbox.

    Implementations execute the ``argv`` vector directly — there is no
    shell interpretation. The facet never owns lifecycle: closing the
    sandbox that vended it is the only teardown.
    """

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        stdin: bytes | None = None,
        timeout_s: float | None = None,
    ) -> CommandResult:
        """Execute ``argv`` and return its captured outcome.

        Args:
            argv: Command vector; ``argv[0]`` is the executable.
            cwd: Working directory relative to the sandbox root; ``None``
                runs at the root.
            env: Extra environment variables layered over the shell's base
                environment for this invocation.
            stdin: Bytes piped to standard input (``None`` closes it).
            timeout_s: Wall-clock limit; ``None`` applies the shell's
                default timeout.

        Raises:
            ValueError: Empty ``argv``, absolute ``cwd``, or non-positive
                ``timeout_s``.
            PermissionError: ``cwd`` escapes the sandbox root.
            FileNotFoundError: ``cwd`` does not exist inside the sandbox.
        """
        ...


def _hygienic_environ() -> dict[str, str]:
    """Host environment minus ``GIT_*`` variables.

    Mirrors the snapshot plumbing's ``git_env`` hygiene: inside git hooks
    the parent git process exports ``GIT_*`` variables that would leak
    host repository state into sandbox commands.
    """
    return {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}


def _as_bytes(value: str | bytes | None) -> bytes:
    if value is None:
        return b""
    if isinstance(value, str):  # pragma: no cover - binary mode yields bytes
        return value.encode("utf-8", errors="replace")
    return value


def _cap(data: bytes, limit: int) -> tuple[bytes, bool]:
    if len(data) <= limit:
        return data, False
    return data[:limit], True


class LocalShell:
    """Run subprocesses rooted at a host directory.

    The base environment is captured at construction: the host environment
    with ``GIT_*`` variables stripped (see :func:`_hygienic_environ`), with
    ``env`` overrides layered on top. Per-invocation ``env`` arguments
    layer over that base. Output is captured fully in memory and capped at
    ``max_output_bytes`` per stream.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        env: Mapping[str, str] | None = None,
        default_timeout_s: float = DEFAULT_SHELL_TIMEOUT_S,
        max_output_bytes: int = MAX_COMMAND_OUTPUT_BYTES,
        clock: MonotonicClock = SYSTEM_CLOCK,
    ) -> None:
        super().__init__()
        if default_timeout_s <= 0:
            msg = f"default_timeout_s must be positive, got {default_timeout_s}"
            raise ValueError(msg)
        if max_output_bytes <= 0:
            msg = f"max_output_bytes must be positive, got {max_output_bytes}"
            raise ValueError(msg)
        self._root = Path(root).resolve()
        self._env: dict[str, str] = _hygienic_environ()
        if env is not None:
            self._env.update(env)
        self._default_timeout_s = default_timeout_s
        self._max_output_bytes = max_output_bytes
        self._clock = clock

    @property
    def root(self) -> str:
        """Directory commands run under (resolved)."""
        return str(self._root)

    def _resolve_cwd(self, cwd: str | None) -> Path:
        """Resolve ``cwd`` inside the root; reject escapes and absolutes.

        Raises:
            ValueError: ``cwd`` is an absolute path.
            PermissionError: ``cwd`` resolves outside the sandbox root.
            FileNotFoundError: ``cwd`` does not exist or is not a directory.
        """
        if cwd is None:
            return self._root
        if Path(cwd).is_absolute():
            msg = f"cwd must be relative to the sandbox root, got {cwd!r}"
            raise ValueError(msg)
        candidate = (self._root / cwd).resolve()
        if candidate != self._root and self._root not in candidate.parents:
            msg = f"cwd escapes the sandbox root: {cwd!r}"
            raise PermissionError(msg)
        if not candidate.is_dir():
            msg = f"cwd does not exist in the sandbox: {cwd!r}"
            raise FileNotFoundError(msg)
        return candidate

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        stdin: bytes | None = None,
        timeout_s: float | None = None,
    ) -> CommandResult:
        """Execute ``argv`` directly (no shell) inside the root.

        See :meth:`Shell.run` for the full contract. Launch failures map to
        shell-conventional exit codes instead of raising: 127 when the
        executable is missing, 126 when it is not executable, 124 with
        ``timed_out=True`` when the timeout kills the command (partial
        output is preserved).
        """
        if not argv:
            msg = "argv must not be empty"
            raise ValueError(msg)
        if timeout_s is not None and timeout_s <= 0:
            msg = f"timeout_s must be positive, got {timeout_s}"
            raise ValueError(msg)
        workdir = self._resolve_cwd(cwd)
        run_env = dict(self._env)
        if env is not None:
            run_env.update(env)
        effective_timeout = (
            timeout_s if timeout_s is not None else self._default_timeout_s
        )

        timed_out = False
        start = self._clock.monotonic()
        try:
            completed = subprocess.run(  # nosec B603
                list(argv),
                cwd=str(workdir),
                env=run_env,
                input=stdin,
                capture_output=True,
                timeout=effective_timeout,
                check=False,
            )
            exit_code = completed.returncode
            raw_stdout, raw_stderr = completed.stdout, completed.stderr
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            exit_code = _EXIT_TIMEOUT
            raw_stdout = _as_bytes(exc.stdout)
            raw_stderr = _as_bytes(exc.stderr)
        except FileNotFoundError:
            exit_code = _EXIT_NOT_FOUND
            raw_stdout = b""
            raw_stderr = f"command not found: {argv[0]}".encode()
        except PermissionError:
            exit_code = _EXIT_NOT_EXECUTABLE
            raw_stdout = b""
            raw_stderr = f"permission denied: {argv[0]}".encode()
        duration_s = self._clock.monotonic() - start

        stdout, out_cut = _cap(raw_stdout, self._max_output_bytes)
        stderr, err_cut = _cap(raw_stderr, self._max_output_bytes)
        return CommandResult(
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            truncated=out_cut or err_cut,
            duration_s=duration_s,
            timed_out=timed_out,
        )
