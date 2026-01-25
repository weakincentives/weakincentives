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

"""Sandboxed Python expression evaluation backed by :mod:`asteval`.

This module provides a secure Python evaluation tool for use in AI agent prompts.
Code is executed in a restricted sandbox with limited builtins (abs, len, max, min,
print, range, round, sum, str) plus math and statistics helpers.

Key components:

- :class:`AstevalSection`: A prompt section that exposes the ``evaluate_python`` tool.
- :class:`AstevalConfig`: Configuration container for AstevalSection settings.
- :class:`EvalParams`: Input parameters for the evaluation tool.
- :class:`EvalResult`: Structured output from the evaluation tool.
- :class:`EvalFileRead` / :class:`EvalFileWrite`: File I/O descriptors for VFS operations.

Example usage::

    from weakincentives.contrib.tools import AstevalConfig, AstevalSection

    config = AstevalConfig(accepts_overrides=True)
    section = AstevalSection(session=session, config=config)

The evaluation tool supports:

- Multi-line Python scripts up to 2,000 characters
- Pre-loading VFS files via the ``reads`` parameter
- Writing files via the ``writes`` parameter or ``write_text()`` helper
- Injecting globals as JSON-encoded strings
- 5-second execution timeout with stdout/stderr capture
"""

from __future__ import annotations

import builtins
import contextlib
import io
import json
import math
import statistics
import threading
from collections.abc import Callable, Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from importlib import import_module
from types import MappingProxyType, ModuleType
from typing import Final, Literal, Protocol, TextIO, cast, override

from weakincentives.filesystem import READ_ENTIRE_FILE, Filesystem

from ...dataclasses import FrozenDataclass
from ...errors import ToolValidationError
from ...prompt.markdown import MarkdownSection
from ...prompt.tool import Tool, ToolContext, ToolExample, ToolResult
from ...runtime.logging import StructuredLogger, get_logger
from ...runtime.session import Session
from .filesystem_memory import InMemoryFilesystem
from .vfs_types import (
    MAX_WRITE_LENGTH as _MAX_WRITE_LENGTH,
    VfsPath,
    ensure_ascii as _ensure_ascii,
    format_path as _format_vfs_path,
    normalize_path as _normalize_vfs_path,
    normalize_segments as _normalize_segments,
)

_LOGGER: StructuredLogger = get_logger(__name__, context={"component": "tools.asteval"})

_MAX_CODE_LENGTH: Final[int] = 2_000
_MAX_STREAM_LENGTH: Final[int] = 4_096
_TIMEOUT_SECONDS: Final[float] = 5.0
_FIRST_PRINTABLE_ASCII: Final[int] = 32  # Space character code point
_MISSING_DEPENDENCY_MESSAGE: Final[str] = (
    "Install weakincentives[asteval] to enable the Python evaluation tool."
)

_SAFE_GLOBALS: Final[Mapping[str, object]] = MappingProxyType(
    {
        "abs": abs,
        "len": len,
        "min": min,
        "max": max,
        "print": print,
        "range": range,
        "round": round,
        "sum": sum,
        "str": str,
        "math": math,
        "statistics": MappingProxyType(
            {
                "mean": statistics.mean,
                "median": statistics.median,
                "pstdev": statistics.pstdev,
                "stdev": statistics.stdev,
                "variance": statistics.variance,
            }
        ),
        "PI": math.pi,
        "TAU": math.tau,
        "E": math.e,
    }
)

_EVAL_TEMPLATE: Final[str] = (
    "Use the Python evaluation tool for quick calculations and one-off scripts.\n"
    "- Scripts run in a sandbox with a narrow set of safe builtins (abs, len, max, min,\n"
    "  print, range, round, sum, str) plus math/statistics helpers. Import statements\n"
    "  and other blocked nodes are stripped, so networking and host filesystem access\n"
    "  are unavailable.\n"
    "- Keep code concise (<=2,000 characters) and avoid control characters other than\n"
    "  newlines or tabs.\n"
    "- Pre-load files via `reads`, or call `read_text(path)` inside code to fetch VFS\n"
    "  files. Paths must be relative, use <=16 segments of <=80 ASCII characters, and\n"
    "  may not target a read and write in the same call.\n"
    "- Stage edits with `write_text(path, content, mode)` or declare them in `writes`.\n"
    "  Content must be ASCII, <=48k characters, and choose from modes create,\n"
    "  overwrite, or append.\n"
    "- Globals accept JSON-encoded strings keyed by valid identifiers. Payloads are\n"
    "  parsed before execution; invalid JSON or names raise a validation error.\n"
    "- Execution stops after five seconds. Stdout/stderr are captured and truncated to\n"
    "  4,096 characters, and the repr of the final expression is returned when present.\n\n"
    "The tool executes multi-line scripts, captures stdout, and returns the repr of the final expression when present:\n"
    "```json\n"
    "{\n"
    '  "name": "evaluate_python",\n'
    '  "arguments": {\n'
    '    "code": "total = 0\\nfor value in range(5):\\n    total += value\\nprint(total)\\ntotal",\n'
    '    "globals": {},\n'
    '    "reads": [],\n'
    '    "writes": []\n'
    "  }\n"
    "}\n"
    "```"
)


def _load_asteval_module() -> ModuleType:
    try:
        return import_module("asteval")
    except ModuleNotFoundError as error:  # pragma: no cover - configuration guard
        raise RuntimeError(_MISSING_DEPENDENCY_MESSAGE) from error


def _str_dict_factory() -> dict[str, str]:
    return {}


def _str_set_factory() -> set[str]:
    return set()


@FrozenDataclass()
class EvalFileRead:
    """Descriptor for a file to read from the virtual filesystem before execution.

    When included in :attr:`EvalParams.reads`, the file content is loaded and made
    available to the sandbox via the ``vfs_reads`` dictionary and also directly
    as a global variable keyed by the file path.

    Attributes:
        path: Relative VFS path to load. Must follow VFS path constraints:
            16 segments max, 80 ASCII characters per segment.

    Example::

        read_spec = EvalFileRead(path=VfsPath(segments=("data", "config.json")))
        params = EvalParams(code="print(vfs_reads['data/config.json'])", reads=(read_spec,))
    """

    path: VfsPath = field(
        metadata={
            "description": (
                "Relative VFS path to load. Contents are injected into "
                "`reads` for convenience."
            )
        }
    )

    def render(self) -> str:
        """Format this read operation as a human-readable string.

        Returns:
            A string in the format ``read <path>``.
        """
        return f"read {_format_vfs_path(self.path)}"


@FrozenDataclass()
class EvalFileWrite:
    """Descriptor for a file to write to the virtual filesystem after execution.

    Writes are applied only if evaluation succeeds (no stderr output). The ``mode``
    parameter controls behavior when the target file exists:

    - ``create``: Write only if file does not exist (skipped otherwise).
    - ``overwrite``: Replace existing content (skipped if file doesn't exist).
    - ``append``: Append to existing content (skipped if file doesn't exist).

    Attributes:
        path: Relative VFS path to create or update.
        content: ASCII text to persist. Maximum 48,000 characters. May contain
            template variables (e.g., ``{variable}``) resolved from the sandbox
            symbol table after execution.
        mode: Write strategy for the file. Defaults to ``"create"``.

    Example::

        write_spec = EvalFileWrite(
            path=VfsPath(segments=("output.txt",)),
            content="Result: {result}",
            mode="create",
        )
        params = EvalParams(code="result = 42", writes=(write_spec,))
    """

    path: VfsPath = field(
        metadata={"description": "Relative VFS path to create or update."}
    )
    content: str = field(
        metadata={
            "description": (
                "ASCII text to persist after execution. Content longer than 48k "
                "characters is rejected."
            )
        }
    )
    mode: Literal["create", "overwrite", "append"] = field(
        default="create",
        metadata={
            "description": (
                "Write strategy for the file: create a new entry, overwrite the "
                "existing content, or append."
            )
        },
    )

    def render(self) -> str:
        """Format this write operation as a human-readable string.

        Returns:
            A string in the format ``<mode> <path> (<size> chars)``.
        """
        size = len(self.content)
        return f"{self.mode} {_format_vfs_path(self.path)} ({size} chars)"


@FrozenDataclass()
class EvalParams:
    """Input parameters for the ``evaluate_python`` tool.

    This dataclass defines the complete input schema for Python evaluation requests.
    It supports code execution with optional file I/O and global variable injection.

    Attributes:
        code: Python script to execute. Maximum 2,000 characters. Control characters
            other than newlines and tabs are rejected.
        globals: Mapping of variable names to JSON-encoded values. Keys must be valid
            Python identifiers. Values are parsed as JSON before injection into the
            sandbox namespace.
        reads: Files to load from the VFS before execution. Contents are available
            via the ``vfs_reads`` dictionary and as direct global variables.
        writes: Files to write after successful execution. Content may include
            template variables resolved from the sandbox symbol table.

    Example::

        params = EvalParams(
            code="total = sum(numbers)\\nprint(total)\\ntotal",
            globals={"numbers": "[1, 2, 3, 4, 5]"},
            reads=(),
            writes=(),
        )

    Note:
        Reads and writes must not target the same path in a single request.
    """

    code: str = field(
        metadata={"description": "Python script to execute (<=2,000 characters)."}
    )
    globals: dict[str, str] = field(
        default_factory=_str_dict_factory,
        metadata={
            "description": (
                "Mapping of global variable names to JSON-encoded strings. The "
                "payload is decoded before execution."
            )
        },
    )
    reads: tuple[EvalFileRead, ...] = field(
        default_factory=tuple,
        metadata={
            "description": (
                "Files to load into the VFS before execution. Each entry is "
                "available to helper utilities."
            )
        },
    )
    writes: tuple[EvalFileWrite, ...] = field(
        default_factory=tuple,
        metadata={
            "description": (
                "Files to write after execution completes. These mirror calls to "
                "`write_text`."
            )
        },
    )


@FrozenDataclass()
class EvalResult:
    """Structured output from the ``evaluate_python`` tool.

    Contains all outputs captured during sandboxed execution: the final expression
    value, stdout/stderr streams, modified globals, and file I/O summaries.

    Attributes:
        value_repr: String representation (via ``repr()``) of the final expression
            result. ``None`` when the code produces no value or errors occur.
        stdout: Captured standard output, truncated to 4,096 characters.
        stderr: Captured standard error including interpreter errors, truncated
            to 4,096 characters. Non-empty stderr indicates evaluation failure.
        globals: Sandbox variables after execution. Includes user-defined variables,
            injected globals, and VFS file contents (keyed as ``vfs:<path>``).
            Values are JSON-encoded strings.
        reads: File read descriptors that were fulfilled during execution.
        writes: File write operations that were applied (empty if execution failed).

    Example result structure::

        EvalResult(
            value_repr="15",
            stdout="Computing sum...\\n",
            stderr="",
            globals={"total": "15", "vfs:data.txt": "1 2 3 4 5"},
            reads=(EvalFileRead(path=VfsPath(segments=("data.txt",))),),
            writes=(),
        )
    """

    value_repr: str | None = field(
        metadata={
            "description": (
                "String representation of the final expression result. Null when "
                "no value was produced."
            )
        }
    )
    stdout: str = field(
        metadata={
            "description": (
                "Captured standard output stream, truncated to 4,096 characters."
            )
        }
    )
    stderr: str = field(
        metadata={
            "description": (
                "Captured standard error stream, truncated to 4,096 characters."
            )
        }
    )
    globals: dict[str, str] = field(
        metadata={
            "description": (
                "JSON-serialisable globals returned from the sandbox after execution."
            )
        }
    )
    reads: tuple[EvalFileRead, ...] = field(
        metadata={"description": "File read requests fulfilled during execution."}
    )
    writes: tuple[EvalFileWrite, ...] = field(
        metadata={"description": "File write operations requested by the code."}
    )

    def render(self) -> str:
        """Format this result as a multi-line human-readable summary.

        Returns:
            A formatted string containing the result value, stdout, stderr,
            file operations, and globals.
        """
        lines: list[str] = ["Python evaluation result:"]
        if self.value_repr is not None:
            lines.append(f"Result: {self.value_repr}")
        lines.append("STDOUT:")
        lines.append(self.stdout or "<empty>")
        lines.append("STDERR:")
        lines.append(self.stderr or "<empty>")
        if self.reads:
            lines.append("Reads:")
            lines.extend(f"- {read.render()}" for read in self.reads)
        if self.writes:
            lines.append("Writes:")
            lines.extend(f"- {write.render()}" for write in self.writes)
        if self.globals:
            lines.append("Globals:")
            lines.extend(f"- {key}={value}" for key, value in self.globals.items())
        return "\n".join(lines)


@FrozenDataclass()
class _AstevalSectionParams:
    """Placeholder params container for the asteval section."""

    pass


@FrozenDataclass()
class AstevalConfig:
    """Configuration container for :class:`AstevalSection`.

    Consolidates all configuration options for AstevalSection in a single
    immutable object. Using a config object avoids accumulating long argument
    lists as the section evolves and makes configuration reusable.

    Attributes:
        accepts_overrides: Whether the ``evaluate_python`` tool accepts parameter
            overrides from the caller. Defaults to ``False``.

    Example::

        from weakincentives.contrib.tools import AstevalConfig, AstevalSection

        config = AstevalConfig(accepts_overrides=True)
        section = AstevalSection(session=session, config=config)

        # Reuse the same config for multiple sections
        section2 = AstevalSection(session=other_session, config=config)
    """

    accepts_overrides: bool = field(
        default=False,
        metadata={"description": "Whether the section accepts parameter overrides."},
    )


def _truncate_stream(text: str) -> str:
    if len(text) <= _MAX_STREAM_LENGTH:
        return text
    suffix = "..."
    keep = _MAX_STREAM_LENGTH - len(suffix)
    return f"{text[:keep]}{suffix}"


def _require_file_from_filesystem(fs: Filesystem, path: VfsPath) -> str:
    """Read file content from filesystem, raising ToolValidationError if not found."""
    normalized = _normalize_vfs_path(path)
    path_str = "/".join(normalized.segments)
    try:
        result = fs.read(path_str)
    except FileNotFoundError:
        raise ToolValidationError(
            "File does not exist in the virtual filesystem."
        ) from None
    else:
        return result.content


def _normalize_code(code: str) -> str:
    if len(code) > _MAX_CODE_LENGTH:
        raise ToolValidationError("Code exceeds maximum length of 2,000 characters.")
    for char in code:
        code_point = ord(char)
        if code_point < _FIRST_PRINTABLE_ASCII and char not in {"\n", "\t"}:
            raise ToolValidationError("Code contains unsupported control characters.")
    return code


def _normalize_write(write: EvalFileWrite) -> EvalFileWrite:
    path = _normalize_vfs_path(write.path)
    content = write.content
    _ensure_ascii(content, "write content")
    if len(content) > _MAX_WRITE_LENGTH:
        raise ToolValidationError(
            "Content exceeds maximum length of 48,000 characters."
        )
    mode = write.mode
    if mode not in {"create", "overwrite", "append"}:
        raise ToolValidationError("Unsupported write mode requested.")
    return EvalFileWrite(path=path, content=content, mode=mode)


def _normalize_reads(reads: Iterable[EvalFileRead]) -> tuple[EvalFileRead, ...]:
    normalized: list[EvalFileRead] = []
    seen: set[tuple[str, ...]] = set()
    for read in reads:
        path = _normalize_vfs_path(read.path)
        key = path.segments
        if key in seen:
            raise ToolValidationError("Duplicate read targets detected.")
        seen.add(key)
        normalized.append(EvalFileRead(path=path))
    return tuple(normalized)


def _normalize_writes(writes: Iterable[EvalFileWrite]) -> tuple[EvalFileWrite, ...]:
    normalized: list[EvalFileWrite] = []
    seen: set[tuple[str, ...]] = set()
    for write in writes:
        normalized_write = _normalize_write(write)
        key = normalized_write.path.segments
        if key in seen:
            raise ToolValidationError("Duplicate write targets detected.")
        seen.add(key)
        normalized.append(normalized_write)
    return tuple(normalized)


def _alias_for_path(path: VfsPath) -> str:
    return "/".join(path.segments)


def _format_value(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return json.dumps(value)
    if isinstance(value, bool) or value is None:
        return json.dumps(value)
    return f"!repr:{value!r}"


def _merge_globals(
    initial: Mapping[str, object], updates: Mapping[str, object]
) -> dict[str, object]:
    merged = dict(initial)
    merged.update(updates)
    return merged


def _summarize_writes(writes: Sequence[EvalFileWrite]) -> str | None:
    if not writes:
        return None

    total = len(writes)
    preview_count = min(3, total)
    preview_paths = ", ".join(
        _alias_for_path(write.path) for write in writes[:preview_count]
    )
    if total > preview_count:
        preview_paths = f"{preview_paths}, +{total - preview_count} more"
    return f"writes={total} file(s): {preview_paths}"


def _apply_writes_to_filesystem(
    fs: Filesystem, writes: Iterable[EvalFileWrite]
) -> None:
    """Apply write operations to the filesystem.

    Writes that fail mode constraints are silently skipped:
    - create mode: skipped if file already exists
    - overwrite/append mode: skipped if file doesn't exist
    """
    for write in writes:
        path_str = "/".join(write.path.segments)
        exists = fs.exists(path_str)

        # Skip writes that fail mode constraints
        if write.mode == "create" and exists:
            continue
        if write.mode in {"overwrite", "append"} and not exists:
            continue

        if write.mode == "append" and exists:
            # Read entire existing content and append
            result = fs.read(path_str, limit=READ_ENTIRE_FILE)
            content = result.content + write.content
            _ = fs.write(path_str, content, mode="overwrite")
        else:
            mode: Literal["create", "overwrite", "append"] = (
                "create" if write.mode == "create" else "overwrite"
            )
            _ = fs.write(path_str, write.content, mode=mode)


def _parse_string_path(path: str) -> VfsPath:
    if not path.strip():
        raise ToolValidationError("Path must be non-empty.")
    return VfsPath(_normalize_segments((path,)))


def _build_eval_globals(
    fs: Filesystem, reads: tuple[EvalFileRead, ...]
) -> dict[str, str]:
    values: dict[str, str] = {}
    for read in reads:
        alias = _alias_for_path(read.path)
        content = _require_file_from_filesystem(fs, read.path)
        values[alias] = content
    return values


def _parse_user_globals(payload: Mapping[str, str]) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for name, encoded in payload.items():
        identifier = name.strip()
        if not identifier:
            raise ToolValidationError("Global variable names must be non-empty.")
        if not identifier.isidentifier():
            raise ToolValidationError(f"Invalid global variable name '{identifier}'.")
        try:
            parsed_value = json.loads(encoded)
        except json.JSONDecodeError as error:
            raise ToolValidationError(
                f"Invalid JSON for global '{identifier}'."
            ) from error
        parsed[identifier] = parsed_value
    return parsed


class InterpreterProtocol(Protocol):
    """Protocol defining the interface for asteval-compatible interpreters.

    This protocol abstracts over the asteval ``Interpreter`` class, enabling
    type-safe usage and testing with mock implementations.

    Attributes:
        symtable: Mutable symbol table containing global variables accessible
            during evaluation.
        node_handlers: Optional mapping of AST node type names to handler
            functions. Used to disable dangerous operations.
        error: List of errors accumulated during evaluation.
    """

    symtable: MutableMapping[str, object]
    node_handlers: MutableMapping[str, object] | None
    error: list[object]

    def eval(self, expression: str) -> object:
        """Evaluate a Python expression or statement.

        Args:
            expression: Python code to evaluate.

        Returns:
            The result of evaluating the expression, or None for statements.
        """
        ...


def _sanitize_interpreter(interpreter: InterpreterProtocol) -> None:
    module = _load_asteval_module()

    for name in getattr(module, "ALL_DISALLOWED", ()):  # pragma: no cover - defensive
        _ = interpreter.symtable.pop(name, None)
    node_handlers = getattr(interpreter, "node_handlers", None)
    if isinstance(
        node_handlers, MutableMapping
    ):  # pragma: no cover - asteval internals
        handlers = cast(MutableMapping[str, object], node_handlers)
        for key in ("Eval", "Exec", "Import", "ImportFrom"):
            _ = handlers.pop(key, None)


def _create_interpreter() -> InterpreterProtocol:
    module = _load_asteval_module()
    interpreter_cls = getattr(module, "Interpreter", None)
    if not callable(interpreter_cls):  # pragma: no cover - defensive guard
        message = _MISSING_DEPENDENCY_MESSAGE
        raise TypeError(message)

    interpreter = cast(
        InterpreterProtocol, interpreter_cls(use_numpy=False, minimal=True)
    )
    interpreter.symtable = dict(_SAFE_GLOBALS)
    _sanitize_interpreter(interpreter)
    return interpreter


def _execute_with_timeout(
    func: Callable[[], object],
) -> tuple[bool, object | None, str]:
    timeout_message = "Execution timed out."
    result_container: dict[str, object | None] = {}
    error_container: dict[str, str] = {"message": ""}
    completed = threading.Event()

    def runner() -> None:
        try:
            result_container["value"] = func()
        except TimeoutError:
            error_container["message"] = timeout_message
        except Exception as error:  # pragma: no cover - forwarded later
            result_container["error"] = error
        finally:
            completed.set()

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    _ = completed.wait(_TIMEOUT_SECONDS)
    if not completed.is_set():
        return True, None, timeout_message
    if "error" in result_container:
        error = cast(Exception, result_container["error"])
        raise error
    return False, result_container.get("value"), error_container["message"]


@dataclass(slots=True)
class _SandboxState:
    interpreter: InterpreterProtocol = field(
        metadata={"description": "Interpreter configured for sandboxed execution."}
    )
    stdout_buffer: io.StringIO = field(
        metadata={"description": "Buffer capturing stdout during evaluation."}
    )
    stderr_buffer: io.StringIO = field(
        metadata={"description": "Buffer capturing stderr during evaluation."}
    )
    symtable: dict[str, object] = field(
        metadata={"description": "Interpreter symbol table used during execution."}
    )
    write_queue: list[EvalFileWrite] = field(
        metadata={"description": "User-declared writes pending template resolution."}
    )
    helper_writes: list[EvalFileWrite] = field(
        metadata={"description": "Writes staged via helper functions during execution."}
    )
    write_targets: set[tuple[str, ...]] = field(
        metadata={"description": "Paths reserved by declared and helper writes."}
    )
    pending_write_attempted: bool = field(
        metadata={"description": "Tracks whether any write helper was invoked."}
    )
    initial_keys: set[str] = field(
        default_factory=_str_set_factory,
        metadata={"description": "Symtable keys captured before evaluation."},
    )


@dataclass(slots=True)
class _ExecutionOutcome:
    value_repr: str | None = field(
        metadata={"description": "repr of the evaluation result when available."}
    )
    stdout: str = field(
        metadata={"description": "Captured stdout text truncated to stream limits."}
    )
    stderr: str = field(
        metadata={"description": "Captured stderr text including interpreter errors."}
    )
    timed_out: bool = field(
        metadata={"description": "Indicates whether evaluation exceeded the timeout."}
    )


@dataclass(slots=True)
class _ResultAssemblyContext:
    reads: tuple[EvalFileRead, ...] = field(
        metadata={"description": "Normalized read descriptors for the current run."}
    )
    final_writes: tuple[EvalFileWrite, ...] = field(
        metadata={"description": "Resolved writes queued after evaluation."}
    )
    sandbox: _SandboxState = field(
        metadata={"description": "Sandbox state containing interpreter metadata."}
    )
    user_globals: Mapping[str, object] = field(
        metadata={"description": "User-provided globals injected into the sandbox."}
    )
    read_globals: Mapping[str, str] = field(
        metadata={"description": "Contents of VFS reads made available to the sandbox."}
    )
    execution: _ExecutionOutcome = field(
        metadata={"description": "Captured interpreter outputs for result assembly."}
    )
    pending_sources: tuple[EvalFileWrite, ...] = field(
        metadata={"description": "Pending writes considered when summarizing output."}
    )
    pending_writes: bool = field(
        metadata={"description": "Indicates whether any writes were requested."}
    )
    base_message: str = field(
        metadata={
            "description": "Base result message before appending write summaries."
        }
    )


class _AstevalToolSuite:
    def __init__(self, *, section: AstevalSection) -> None:
        super().__init__()
        self._section = section

    def run(
        self, params: EvalParams, *, context: ToolContext
    ) -> ToolResult[EvalResult]:
        del context  # Filesystem accessed via section
        fs = self._section._filesystem  # pyright: ignore[reportPrivateUsage]
        code = _normalize_code(params.code)
        reads = _normalize_reads(params.reads)
        writes = _normalize_writes(params.writes)
        read_paths = {read.path.segments for read in reads}
        write_paths = {write.path.segments for write in writes}
        if read_paths & write_paths:
            raise ToolValidationError("Reads and writes must not target the same path.")

        read_globals = _build_eval_globals(fs, reads)
        user_globals = _parse_user_globals(params.globals)

        sandbox = self._initialize_sandbox(
            fs=fs,
            reads=reads,
            writes=writes,
            read_globals=read_globals,
            user_globals=user_globals,
        )
        execution = self._execute_interpreter(
            interpreter=sandbox.interpreter,
            code=code,
            stdout_buffer=sandbox.stdout_buffer,
            stderr_buffer=sandbox.stderr_buffer,
        )
        (
            final_writes,
            message,
            pending_sources,
            pending_writes,
        ) = self._resolve_writes(
            symtable=sandbox.symtable,
            param_writes=tuple(sandbox.write_queue),
            helper_writes=tuple(sandbox.helper_writes),
            pending_write_attempted=sandbox.pending_write_attempted,
            execution=execution,
        )

        # Apply writes to filesystem if evaluation succeeded
        if final_writes and not execution.stderr:
            _apply_writes_to_filesystem(fs, final_writes)

        result = self._assemble_result(
            _ResultAssemblyContext(
                reads=reads,
                final_writes=final_writes,
                sandbox=sandbox,
                user_globals=user_globals,
                read_globals=read_globals,
                execution=execution,
                pending_sources=pending_sources,
                pending_writes=pending_writes,
                base_message=message,
            )
        )

        _LOGGER.debug(
            "Asteval evaluation completed.",
            event="asteval_run",
            context={
                "stdout_len": len(execution.stdout),
                "stderr_len": len(execution.stderr),
                "write_count": len(final_writes),
                "code_preview": code[:200],
            },
        )

        return result

    @staticmethod
    def _create_sandbox_print(
        state: _SandboxState,
    ) -> Callable[..., None]:
        """Create a sandbox-safe print function."""
        builtin_print = builtins.print

        def sandbox_print(
            *args: object,
            sep: object | None = " ",
            end: object | None = "\n",
            file: TextIO | None = None,
            flush: bool = False,
        ) -> None:
            if sep is not None and not isinstance(sep, str):
                raise TypeError("sep must be None or a string.")
            if end is not None and not isinstance(end, str):
                raise TypeError("end must be None or a string.")
            actual_sep = " " if sep is None else str(sep)
            actual_end = "\n" if end is None else str(end)
            if file is not None:  # pragma: no cover - requires custom injected writer
                builtin_print(
                    *args, sep=actual_sep, end=actual_end, file=file, flush=flush
                )
                return
            text = actual_sep.join(str(arg) for arg in args)
            _ = state.stdout_buffer.write(text)
            _ = state.stdout_buffer.write(actual_end)
            if flush:
                _ = state.stdout_buffer.flush()

        return sandbox_print

    @staticmethod
    def _create_write_text(
        state: _SandboxState, read_paths: set[tuple[str, ...]]
    ) -> Callable[[str, str, str], None]:
        """Create a sandbox-safe write_text function."""

        def write_text(path: str, content: str, mode: str = "create") -> None:
            state.pending_write_attempted = True
            normalized_path = _normalize_vfs_path(_parse_string_path(path))
            helper_write = _normalize_write(
                EvalFileWrite(
                    path=normalized_path,
                    content=content,
                    mode=cast(Literal["create", "overwrite", "append"], mode),
                )
            )
            key = helper_write.path.segments
            if key in read_paths:
                raise ToolValidationError(
                    "Writes queued during execution must not target read paths."
                )
            if key in state.write_targets:
                raise ToolValidationError("Duplicate write targets detected.")
            state.write_targets.add(key)
            state.helper_writes.append(helper_write)

        return write_text

    @staticmethod
    def _initialize_sandbox(
        *,
        fs: Filesystem,
        reads: tuple[EvalFileRead, ...],
        writes: tuple[EvalFileWrite, ...],
        read_globals: Mapping[str, str],
        user_globals: Mapping[str, object],
    ) -> _SandboxState:
        interpreter = _create_interpreter()
        write_queue: list[EvalFileWrite] = list(writes)
        state = _SandboxState(
            interpreter=interpreter,
            stdout_buffer=io.StringIO(),
            stderr_buffer=io.StringIO(),
            symtable=cast(dict[str, object], interpreter.symtable),
            write_queue=write_queue,
            helper_writes=[],
            write_targets={write.path.segments for write in write_queue},
            pending_write_attempted=bool(write_queue),
        )
        read_paths = {read.path.segments for read in reads}

        def read_text(path: str) -> str:
            normalized = _normalize_vfs_path(_parse_string_path(path))
            return _require_file_from_filesystem(fs, normalized)

        symtable = state.symtable
        symtable.update(_merge_globals(read_globals, user_globals))
        symtable["vfs_reads"] = dict(read_globals)
        symtable["read_text"] = read_text
        symtable["write_text"] = _AstevalToolSuite._create_write_text(state, read_paths)
        symtable["print"] = _AstevalToolSuite._create_sandbox_print(state)
        state.initial_keys = set(symtable)
        return state

    @staticmethod
    def _execute_interpreter(
        *,
        interpreter: InterpreterProtocol,
        code: str,
        stdout_buffer: io.StringIO,
        stderr_buffer: io.StringIO,
    ) -> _ExecutionOutcome:
        captured_errors: list[str] = []
        value_repr: str | None = None
        stderr_text = ""
        timed_out = False
        try:
            with (
                contextlib.redirect_stdout(stdout_buffer),
                contextlib.redirect_stderr(stderr_buffer),
            ):
                interpreter.error = []

                def runner() -> object:
                    return interpreter.eval(code)

                timed_out, result, timeout_error = _execute_with_timeout(runner)
                if timed_out:
                    stderr_text = timeout_error
                elif interpreter.error:
                    captured_errors.extend(str(err) for err in interpreter.error)
                if not timed_out and not captured_errors and not stderr_text:
                    value_repr = None if result is None else repr(result)
        except ToolValidationError:  # pragma: no cover - interpreter wraps tool errors
            raise
        except Exception as error:  # pragma: no cover - runtime exception
            captured_errors.append(str(error))
        stdout = _truncate_stream(stdout_buffer.getvalue())
        stderr_raw = (
            stderr_text or "\n".join(captured_errors) or stderr_buffer.getvalue()
        )
        stderr = _truncate_stream(stderr_raw)
        return _ExecutionOutcome(
            value_repr=value_repr,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
        )

    @staticmethod
    def _resolve_writes(
        *,
        symtable: Mapping[str, object],
        param_writes: tuple[EvalFileWrite, ...],
        helper_writes: tuple[EvalFileWrite, ...],
        pending_write_attempted: bool,
        execution: _ExecutionOutcome,
    ) -> tuple[
        tuple[EvalFileWrite, ...],
        str,
        tuple[EvalFileWrite, ...],
        bool,
    ]:
        pending_writes = pending_write_attempted or bool(helper_writes)
        if execution.stderr and not execution.value_repr:
            final_writes: tuple[EvalFileWrite, ...] = ()
            if pending_writes:
                message = (
                    "Evaluation failed; pending file writes were discarded. "
                    "Review stderr details in the payload."
                )
            else:
                message = "Evaluation failed; review stderr details in the payload."
            pending_sources = final_writes or param_writes + helper_writes
            return final_writes, message, pending_sources, pending_writes

        format_context = {
            key: value for key, value in symtable.items() if not key.startswith("_")
        }
        resolved_param_writes: list[EvalFileWrite] = []
        for write in param_writes:
            try:
                resolved_content = write.content.format_map(format_context)
            except KeyError as error:
                missing = error.args[0]
                raise ToolValidationError(
                    f"Missing template variable '{missing}' in write request."
                ) from error
            resolved_param_writes.append(
                _normalize_write(
                    EvalFileWrite(
                        path=write.path,
                        content=resolved_content,
                        mode=write.mode,
                    )
                )
            )
        final_writes = tuple(resolved_param_writes + list(helper_writes))
        seen_targets: set[tuple[str, ...]] = set()
        for write in final_writes:
            key = write.path.segments
            if key in seen_targets:
                raise ToolValidationError(
                    "Duplicate write targets detected."
                )  # pragma: no cover - upstream checks prevent duplicates
            seen_targets.add(key)
        if final_writes:
            message = (
                "Evaluation succeeded with "
                f"{len(final_writes)} pending file write"
                f"{'s' if len(final_writes) != 1 else ''}."
            )
        else:
            message = "Evaluation succeeded without pending file writes."
        pending_sources = final_writes or param_writes + helper_writes
        return final_writes, message, pending_sources, pending_writes

    @staticmethod
    def _assemble_result(context: _ResultAssemblyContext) -> ToolResult[EvalResult]:
        summary = _summarize_writes(context.pending_sources)
        message = context.base_message
        if context.pending_writes and summary:
            message = f"{message} {summary}"

        globals_payload: dict[str, str] = {}
        symtable = context.sandbox.symtable
        visible_keys = {
            key
            for key in symtable
            if key not in context.sandbox.initial_keys and not key.startswith("_")
        }
        visible_keys.update(context.user_globals.keys())
        for key in visible_keys:
            globals_payload[key] = _format_value(symtable.get(key))
        globals_payload.update(
            {f"vfs:{alias}": content for alias, content in context.read_globals.items()}
        )

        result = EvalResult(
            value_repr=context.execution.value_repr,
            stdout=context.execution.stdout,
            stderr=context.execution.stderr,
            globals=globals_payload,
            reads=context.reads,
            writes=context.final_writes,
        )
        return ToolResult(message=message, value=result)


def normalize_eval_reads(reads: Iterable[EvalFileRead]) -> tuple[EvalFileRead, ...]:
    """Normalize and validate a collection of file read descriptors.

    Validates each path against VFS constraints and ensures no duplicate targets.

    Args:
        reads: Iterable of :class:`EvalFileRead` instances to normalize.

    Returns:
        A tuple of normalized read descriptors with validated paths.

    Raises:
        ToolValidationError: If paths are invalid or duplicates are detected.
    """
    return _normalize_reads(reads)


def normalize_eval_writes(
    writes: Iterable[EvalFileWrite],
) -> tuple[EvalFileWrite, ...]:
    """Normalize and validate a collection of file write descriptors.

    Validates paths, content (ASCII-only, max 48k chars), and modes. Rejects
    duplicate write targets.

    Args:
        writes: Iterable of :class:`EvalFileWrite` instances to normalize.

    Returns:
        A tuple of normalized write descriptors with validated paths and content.

    Raises:
        ToolValidationError: If validation fails for any write descriptor.
    """
    return _normalize_writes(writes)


def normalize_eval_write(write: EvalFileWrite) -> EvalFileWrite:
    """Normalize and validate a single file write descriptor.

    Validates the path against VFS constraints, ensures content is ASCII-only
    and within the 48,000 character limit, and verifies the mode is valid.

    Args:
        write: The :class:`EvalFileWrite` instance to normalize.

    Returns:
        A new :class:`EvalFileWrite` with normalized path.

    Raises:
        ToolValidationError: If the path, content, or mode is invalid.
    """
    return _normalize_write(write)


def parse_eval_globals(payload: Mapping[str, str]) -> dict[str, object]:
    """Parse a mapping of JSON-encoded global variables.

    Validates that each key is a valid Python identifier and each value is
    valid JSON.

    Args:
        payload: Mapping of variable names to JSON-encoded string values.

    Returns:
        Dictionary mapping variable names to their parsed Python values.

    Raises:
        ToolValidationError: If any name is not a valid identifier or any
            value is not valid JSON.

    Example::

        globals_dict = parse_eval_globals({"numbers": "[1, 2, 3]", "name": '"Alice"'})
        # Returns: {"numbers": [1, 2, 3], "name": "Alice"}
    """
    return _parse_user_globals(payload)


def alias_for_eval_path(path: VfsPath) -> str:
    """Convert a VFS path to its string alias form.

    The alias is used as the key in ``vfs_reads`` and when reporting file
    operations.

    Args:
        path: The :class:`VfsPath` to convert.

    Returns:
        The path segments joined by forward slashes (e.g., ``"data/config.json"``).

    Example::

        path = VfsPath(segments=("data", "config.json"))
        alias = alias_for_eval_path(path)  # Returns "data/config.json"
    """
    return _alias_for_path(path)


def summarize_eval_writes(writes: Sequence[EvalFileWrite]) -> str | None:
    """Generate a human-readable summary of pending write operations.

    Creates a compact summary showing the count and paths of writes, suitable
    for including in tool result messages.

    Args:
        writes: Sequence of :class:`EvalFileWrite` instances to summarize.

    Returns:
        A summary string like ``"writes=3 file(s): a.txt, b.txt, c.txt"``,
        or ``None`` if the sequence is empty.

    Example::

        writes = [
            EvalFileWrite(path=VfsPath(segments=("a.txt",)), content="..."),
            EvalFileWrite(path=VfsPath(segments=("b.txt",)), content="..."),
        ]
        summary = summarize_eval_writes(writes)
        # Returns: "writes=2 file(s): a.txt, b.txt"
    """
    return _summarize_writes(writes)


class AstevalSection(MarkdownSection[_AstevalSectionParams]):
    """Prompt section that provides the ``evaluate_python`` sandboxed execution tool.

    This section exposes a tool for running Python code in a restricted sandbox
    backed by the :mod:`asteval` library. The sandbox provides:

    - Limited builtins: ``abs``, ``len``, ``max``, ``min``, ``print``, ``range``,
      ``round``, ``sum``, ``str``, plus ``math`` and ``statistics`` modules.
    - VFS integration via ``read_text()`` and ``write_text()`` helpers.
    - 5-second execution timeout with stdout/stderr capture.
    - Global variable injection via JSON-encoded payloads.

    Example::

        from weakincentives.contrib.tools import AstevalConfig, AstevalSection

        config = AstevalConfig(accepts_overrides=True)
        section = AstevalSection(session=session, config=config)

        # Add to a prompt template
        prompt = PromptTemplate(
            ns="myapp",
            key="computation",
            sections=(section,),
        )

    The section can share a filesystem with :class:`VfsToolsSection` by passing
    the same ``filesystem`` instance to both.

    Attributes:
        session: The session instance used by this section.
        filesystem: The virtual filesystem for file I/O operations.
    """

    def __init__(
        self,
        *,
        session: Session,
        config: AstevalConfig | None = None,
        filesystem: Filesystem | None = None,
        accepts_overrides: bool = False,
    ) -> None:
        """Initialize the asteval section.

        Args:
            session: The session instance for state management.
            config: Optional configuration container. Takes precedence over
                individual parameters when provided.
            filesystem: Optional filesystem for VFS operations. Defaults to a
                new :class:`InMemoryFilesystem` if not provided. Pass the same
                instance to multiple sections to share file state.
            accepts_overrides: Whether the tool accepts parameter overrides.
                Ignored if ``config`` is provided.
        """
        # Resolve config - explicit config takes precedence
        if config is not None:
            resolved_accepts_overrides = config.accepts_overrides
        else:
            resolved_accepts_overrides = accepts_overrides

        self._session = session
        # Use provided filesystem or create a new one
        self._filesystem = (
            filesystem if filesystem is not None else InMemoryFilesystem()
        )

        # Store config for cloning
        self._config = AstevalConfig(accepts_overrides=resolved_accepts_overrides)

        tool_suite = _AstevalToolSuite(section=self)
        tool = Tool[EvalParams, EvalResult](
            name="evaluate_python",
            description=(
                "Run a short Python expression or script in a sandbox. Supports "
                "preloading VFS files, staging writes, and returning captured "
                "stdout, stderr, and result data."
            ),
            handler=tool_suite.run,
            accepts_overrides=resolved_accepts_overrides,
            examples=(
                ToolExample(
                    description=(
                        "Sum contents of a staged VFS file and capture stdout."
                    ),
                    input=EvalParams(
                        code=(
                            "total = sum(int(value) for value in "
                            "read_text('numbers.txt').split())\nprint(total)\n"
                            "total"
                        ),
                        globals={},
                        reads=(
                            EvalFileRead(
                                path=VfsPath(segments=("numbers.txt",)),
                            ),
                        ),
                        writes=(),
                    ),
                    output=EvalResult(
                        value_repr="6",
                        stdout="6\n",
                        stderr="",
                        globals={
                            "total": "6",
                            "vfs:numbers.txt": "1\n2\n3\n",
                        },
                        reads=(EvalFileRead(path=VfsPath(segments=("numbers.txt",))),),
                        writes=(),
                    ),
                ),
            ),
        )
        super().__init__(
            title="Python Evaluation Tool",
            key="tools.asteval",
            template=_EVAL_TEMPLATE,
            default_params=_AstevalSectionParams(),
            tools=(tool,),
            accepts_overrides=resolved_accepts_overrides,
        )

    @property
    def session(self) -> Session:
        """The session instance used by this section."""
        return self._session

    @property
    def filesystem(self) -> Filesystem:
        """The virtual filesystem used for file I/O operations.

        Returns the filesystem instance that was provided during construction
        or the auto-created :class:`InMemoryFilesystem`. This filesystem is
        shared with the ``evaluate_python`` tool for ``read_text()`` and
        ``write_text()`` operations.
        """
        return self._filesystem

    @override
    def clone(self, **kwargs: object) -> AstevalSection:
        """Create a copy of this section with a new session.

        This method is used internally when a prompt template creates section
        instances for a specific session context.

        Args:
            **kwargs: Must include ``session`` (required) as a :class:`Session`
                instance. May include ``filesystem`` as a :class:`Filesystem`
                instance to share file state; if not provided, the current
                filesystem is reused.

        Returns:
            A new :class:`AstevalSection` with the provided session and the
            same configuration as this section.

        Raises:
            TypeError: If ``session`` is missing or not a Session, or if
                ``filesystem`` is provided but not a Filesystem instance.
        """
        session = kwargs.get("session")
        if not isinstance(session, Session):
            msg = "session is required to clone AstevalSection."
            raise TypeError(msg)
        # Use provided filesystem if given, otherwise keep the current one.
        # This allows sharing a filesystem across sections (e.g., with VfsToolsSection).
        filesystem = kwargs.get("filesystem")
        if filesystem is not None and not isinstance(filesystem, Filesystem):
            msg = "filesystem must be a Filesystem instance."
            raise TypeError(msg)
        return AstevalSection(
            session=session,
            config=self._config,
            filesystem=filesystem if filesystem is not None else self._filesystem,
        )


__all__ = [
    "AstevalConfig",
    "AstevalSection",
    "EvalFileRead",
    "EvalFileWrite",
    "EvalParams",
    "EvalResult",
    "alias_for_eval_path",
    "normalize_eval_reads",
    "normalize_eval_write",
    "normalize_eval_writes",
    "parse_eval_globals",
    "summarize_eval_writes",
]
