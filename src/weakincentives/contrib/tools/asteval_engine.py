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

"""Asteval execution engine for sandboxed Python evaluation.

This module contains the core evaluation logic extracted from the asteval
tool suite, enabling isolated testing of the evaluation engine without
the prompt section machinery.
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
from typing import Final, Literal, Protocol, TextIO, cast

from ...errors import ToolValidationError
from ...prompt.tool import ToolContext, ToolResult
from ...runtime.logging import StructuredLogger, get_logger
from .asteval_types import EvalFileRead, EvalFileWrite, EvalParams, EvalResult
from .filesystem import READ_ENTIRE_FILE, Filesystem
from .vfs_types import (
    MAX_WRITE_LENGTH as _MAX_WRITE_LENGTH,
    VfsPath,
    ensure_ascii as _ensure_ascii,
    normalize_path as _normalize_vfs_path,
    normalize_segments as _normalize_segments,
)

_LOGGER: StructuredLogger = get_logger(
    __name__, context={"component": "tools.asteval_engine"}
)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

MAX_CODE_LENGTH: Final[int] = 2_000
MAX_STREAM_LENGTH: Final[int] = 4_096
TIMEOUT_SECONDS: Final[float] = 5.0
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

# -----------------------------------------------------------------------------
# Interpreter Protocol and Factory
# -----------------------------------------------------------------------------


def load_asteval_module() -> ModuleType:
    try:
        return import_module("asteval")
    except ModuleNotFoundError as error:  # pragma: no cover - configuration guard
        raise RuntimeError(_MISSING_DEPENDENCY_MESSAGE) from error


class InterpreterProtocol(Protocol):
    symtable: MutableMapping[str, object]
    node_handlers: MutableMapping[str, object] | None
    error: list[object]

    def eval(self, expression: str) -> object: ...


def _sanitize_interpreter(interpreter: InterpreterProtocol) -> None:
    module = load_asteval_module()

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
    module = load_asteval_module()
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


# -----------------------------------------------------------------------------
# Timeout Execution
# -----------------------------------------------------------------------------


def execute_with_timeout(
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
    _ = completed.wait(TIMEOUT_SECONDS)
    if not completed.is_set():
        return True, None, timeout_message
    if "error" in result_container:
        error = cast(Exception, result_container["error"])
        raise error
    return False, result_container.get("value"), error_container["message"]


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def _str_set_factory() -> set[str]:
    return set()


def truncate_stream(text: str) -> str:
    if len(text) <= MAX_STREAM_LENGTH:
        return text
    suffix = "..."
    keep = MAX_STREAM_LENGTH - len(suffix)
    return f"{text[:keep]}{suffix}"


def require_file_from_filesystem(fs: Filesystem, path: VfsPath) -> str:
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


def normalize_code(code: str) -> str:
    if len(code) > MAX_CODE_LENGTH:
        raise ToolValidationError("Code exceeds maximum length of 2,000 characters.")
    for char in code:
        code_point = ord(char)
        if code_point < _FIRST_PRINTABLE_ASCII and char not in {"\n", "\t"}:
            raise ToolValidationError("Code contains unsupported control characters.")
    return code


def alias_for_path(path: VfsPath) -> str:
    return "/".join(path.segments)


def format_value(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return json.dumps(value)
    if isinstance(value, bool) or value is None:
        return json.dumps(value)
    return f"!repr:{value!r}"


def merge_globals(
    initial: Mapping[str, object], updates: Mapping[str, object]
) -> dict[str, object]:
    merged = dict(initial)
    merged.update(updates)
    return merged


def parse_string_path(path: str) -> VfsPath:
    if not path.strip():
        raise ToolValidationError("Path must be non-empty.")
    return VfsPath(_normalize_segments((path,)))


def parse_user_globals(payload: Mapping[str, str]) -> dict[str, object]:
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


# -----------------------------------------------------------------------------
# Sandbox State Dataclasses
# -----------------------------------------------------------------------------


@dataclass(slots=True)
class SandboxState:
    """Mutable state container for sandbox execution.

    Tracks interpreter state, I/O buffers, and pending file operations
    during a single evaluation run.
    """

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
class ExecutionOutcome:
    """Captured result of interpreter execution.

    Contains the repr of the result value (if any), captured stdout/stderr
    streams, and timeout status.
    """

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
class ResultAssemblyContext:
    """Context for assembling the final ToolResult.

    Aggregates all data needed to construct the evaluation result,
    including reads, writes, globals, and execution outcome.
    """

    reads: tuple[EvalFileRead, ...] = field(
        metadata={"description": "Normalized read descriptors for the current run."}
    )
    final_writes: tuple[EvalFileWrite, ...] = field(
        metadata={"description": "Resolved writes queued after evaluation."}
    )
    sandbox: SandboxState = field(
        metadata={"description": "Sandbox state containing interpreter metadata."}
    )
    user_globals: Mapping[str, object] = field(
        metadata={"description": "User-provided globals injected into the sandbox."}
    )
    read_globals: Mapping[str, str] = field(
        metadata={"description": "Contents of VFS reads made available to the sandbox."}
    )
    execution: ExecutionOutcome = field(
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


# -----------------------------------------------------------------------------
# Read/Write Normalization
# -----------------------------------------------------------------------------


def normalize_write(write: EvalFileWrite) -> EvalFileWrite:
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


def normalize_reads(reads: Iterable[EvalFileRead]) -> tuple[EvalFileRead, ...]:
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


def normalize_writes(
    writes: Iterable[EvalFileWrite],
) -> tuple[EvalFileWrite, ...]:
    normalized: list[EvalFileWrite] = []
    seen: set[tuple[str, ...]] = set()
    for write in writes:
        normalized_write = normalize_write(write)
        key = normalized_write.path.segments
        if key in seen:
            raise ToolValidationError("Duplicate write targets detected.")
        seen.add(key)
        normalized.append(normalized_write)
    return tuple(normalized)


def summarize_writes(writes: Sequence[EvalFileWrite]) -> str | None:
    if not writes:
        return None

    total = len(writes)
    preview_count = min(3, total)
    preview_paths = ", ".join(
        alias_for_path(write.path) for write in writes[:preview_count]
    )
    if total > preview_count:
        preview_paths = f"{preview_paths}, +{total - preview_count} more"
    return f"writes={total} file(s): {preview_paths}"


def apply_writes_to_filesystem(fs: Filesystem, writes: Iterable[EvalFileWrite]) -> None:
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


def build_eval_globals(
    fs: Filesystem, reads: tuple[EvalFileRead, ...]
) -> dict[str, str]:
    values: dict[str, str] = {}
    for read in reads:
        alias = alias_for_path(read.path)
        content = require_file_from_filesystem(fs, read.path)
        values[alias] = content
    return values


# -----------------------------------------------------------------------------
# Asteval Tool Suite (Engine)
# -----------------------------------------------------------------------------


class AstevalToolSuite:
    """Execution engine for the asteval Python evaluation tool.

    This class encapsulates the sandboxed evaluation logic, handling:
    - Sandbox initialization with restricted globals
    - Code execution with timeout
    - Write resolution and filesystem updates
    - Result assembly

    The engine is designed to be testable in isolation from the
    prompt section infrastructure.
    """

    def __init__(self, *, filesystem: Filesystem) -> None:
        super().__init__()
        self._filesystem = filesystem

    def run(
        self, params: EvalParams, *, context: ToolContext
    ) -> ToolResult[EvalResult]:
        del context  # Filesystem provided via constructor
        fs = self._filesystem
        code = normalize_code(params.code)
        reads = normalize_reads(params.reads)
        writes = normalize_writes(params.writes)
        read_paths = {read.path.segments for read in reads}
        write_paths = {write.path.segments for write in writes}
        if read_paths & write_paths:
            raise ToolValidationError("Reads and writes must not target the same path.")

        read_globals = build_eval_globals(fs, reads)
        user_globals = parse_user_globals(params.globals)

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
            apply_writes_to_filesystem(fs, final_writes)

        result = self._assemble_result(
            ResultAssemblyContext(
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
        state: SandboxState,
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
        state: SandboxState, read_paths: set[tuple[str, ...]]
    ) -> Callable[[str, str, str], None]:
        """Create a sandbox-safe write_text function."""

        def write_text(path: str, content: str, mode: str = "create") -> None:
            state.pending_write_attempted = True
            normalized_path = _normalize_vfs_path(parse_string_path(path))
            helper_write = normalize_write(
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
    ) -> SandboxState:
        interpreter = _create_interpreter()
        write_queue: list[EvalFileWrite] = list(writes)
        state = SandboxState(
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
            normalized = _normalize_vfs_path(parse_string_path(path))
            return require_file_from_filesystem(fs, normalized)

        symtable = state.symtable
        symtable.update(merge_globals(read_globals, user_globals))
        symtable["vfs_reads"] = dict(read_globals)
        symtable["read_text"] = read_text
        symtable["write_text"] = AstevalToolSuite._create_write_text(state, read_paths)
        symtable["print"] = AstevalToolSuite._create_sandbox_print(state)
        state.initial_keys = set(symtable)
        return state

    @staticmethod
    def _execute_interpreter(
        *,
        interpreter: InterpreterProtocol,
        code: str,
        stdout_buffer: io.StringIO,
        stderr_buffer: io.StringIO,
    ) -> ExecutionOutcome:
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

                timed_out, result, timeout_error = execute_with_timeout(runner)
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
        stdout = truncate_stream(stdout_buffer.getvalue())
        stderr_raw = (
            stderr_text or "\n".join(captured_errors) or stderr_buffer.getvalue()
        )
        stderr = truncate_stream(stderr_raw)
        return ExecutionOutcome(
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
        execution: ExecutionOutcome,
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
                normalize_write(
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
    def _assemble_result(
        context: ResultAssemblyContext,
    ) -> ToolResult[EvalResult]:
        summary = summarize_writes(context.pending_sources)
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
            globals_payload[key] = format_value(symtable.get(key))
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


__all__ = [
    "AstevalToolSuite",
    "ExecutionOutcome",
    "InterpreterProtocol",
    "ResultAssemblyContext",
    "SandboxState",
]
