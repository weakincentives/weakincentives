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

"""Sandboxed Python expression evaluation backed by :mod:`asteval`."""

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
from datetime import UTC, datetime
from importlib import import_module
from types import MappingProxyType, ModuleType
from typing import Final, Literal, Protocol, TextIO, cast

from ..prompt.markdown import MarkdownSection
from ..prompt.tool import Tool, ToolContext, ToolResult
from ..runtime.logging import StructuredLogger, get_logger
from ..runtime.session import (
    ReducerContextProtocol,
    ReducerEvent,
    Session,
    TypedReducer,
    select_latest,
)
from ._context import ensure_context_uses_session
from .errors import ToolValidationError
from .vfs import VfsFile, VfsPath, VirtualFileSystem

_LOGGER: StructuredLogger = get_logger(__name__, context={"component": "tools.asteval"})

_MAX_CODE_LENGTH: Final[int] = 2_000
_MAX_STREAM_LENGTH: Final[int] = 4_096
_MAX_WRITE_LENGTH: Final[int] = 48_000
_MAX_PATH_DEPTH: Final[int] = 16
_MAX_SEGMENT_LENGTH: Final[int] = 80
_ASCII: Final[str] = "ascii"
_TIMEOUT_SECONDS: Final[float] = 5.0
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


@dataclass(slots=True, frozen=True)
class EvalFileRead:
    """File that should be read from the virtual filesystem before execution."""

    path: VfsPath = field(
        metadata={
            "description": (
                "Relative VFS path to load. Contents are injected into "
                "`reads` for convenience."
            )
        }
    )


@dataclass(slots=True, frozen=True)
class EvalFileWrite:
    """File that should be written back to the virtual filesystem."""

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


@dataclass(slots=True, frozen=True)
class EvalParams:
    """Parameter payload passed to the Python evaluation tool."""

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


@dataclass(slots=True, frozen=True)
class EvalResult:
    """Structured result produced by the Python evaluation tool."""

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


@dataclass(slots=True, frozen=True)
class _AstevalSectionParams:
    """Placeholder params container for the asteval section."""

    pass


def _now() -> datetime:
    value = datetime.now(UTC)
    microsecond = value.microsecond - value.microsecond % 1000
    return value.replace(microsecond=microsecond, tzinfo=UTC)


def _truncate_stream(text: str) -> str:
    if len(text) <= _MAX_STREAM_LENGTH:
        return text
    suffix = "..."
    keep = _MAX_STREAM_LENGTH - len(suffix)
    return f"{text[:keep]}{suffix}"


def _ensure_ascii(value: str, label: str) -> None:
    try:
        _ = value.encode(_ASCII)
    except UnicodeEncodeError as error:  # pragma: no cover - defensive guard
        raise ToolValidationError(f"{label} must be ASCII text.") from error


def _normalize_segments(raw_segments: Iterable[str]) -> tuple[str, ...]:
    segments: list[str] = []
    for raw_segment in raw_segments:
        stripped = raw_segment.strip()
        if not stripped:
            continue
        if stripped.startswith("/"):
            raise ToolValidationError("Absolute paths are not allowed in the VFS.")
        for piece in stripped.split("/"):
            if not piece:
                continue
            if piece in {".", ".."}:
                raise ToolValidationError("Path segments may not include '.' or '..'.")
            _ensure_ascii(piece, "path segment")
            if len(piece) > _MAX_SEGMENT_LENGTH:
                raise ToolValidationError(
                    "Path segments must be 80 characters or fewer."
                )
            segments.append(piece)
    if len(segments) > _MAX_PATH_DEPTH:
        raise ToolValidationError("Path depth exceeds the allowed limit (16 segments).")
    return tuple(segments)


def _normalize_vfs_path(path: VfsPath) -> VfsPath:
    return VfsPath(_normalize_segments(path.segments))


def _require_file(snapshot: VirtualFileSystem, path: VfsPath) -> VfsFile:
    normalized = _normalize_vfs_path(path)
    for file in snapshot.files:
        if file.path.segments == normalized.segments:
            return file
    raise ToolValidationError("File does not exist in the virtual filesystem.")


def _normalize_code(code: str) -> str:
    if len(code) > _MAX_CODE_LENGTH:
        raise ToolValidationError("Code exceeds maximum length of 2,000 characters.")
    for char in code:
        code_point = ord(char)
        if code_point < 32 and char not in {"\n", "\t"}:
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


def _apply_writes(
    snapshot: VirtualFileSystem, writes: Iterable[EvalFileWrite]
) -> VirtualFileSystem:
    files = list(snapshot.files)
    timestamp = _now()
    for write in writes:
        existing_index = next(
            (index for index, file in enumerate(files) if file.path == write.path),
            None,
        )
        existing = files[existing_index] if existing_index is not None else None
        if write.mode == "create" and existing is not None:
            raise ToolValidationError("File already exists; use overwrite or append.")
        if write.mode in {"overwrite", "append"} and existing is None:
            raise ToolValidationError("File does not exist for the requested mode.")
        if write.mode == "append" and existing is not None:
            content = existing.content + write.content
            created_at = existing.created_at
            version = existing.version + 1
        elif existing is not None:
            content = write.content
            created_at = existing.created_at
            version = existing.version + 1
        else:
            content = write.content
            created_at = timestamp
            version = 1
        size_bytes = len(content.encode("utf-8"))
        updated = VfsFile(
            path=write.path,
            content=content,
            encoding="utf-8",
            size_bytes=size_bytes,
            version=version,
            created_at=created_at,
            updated_at=timestamp,
        )
        if existing_index is not None:
            _ = files.pop(existing_index)
        files.append(updated)
    files.sort(key=lambda file: file.path.segments)
    return VirtualFileSystem(files=tuple(files))


def _parse_string_path(path: str) -> VfsPath:
    if not path.strip():
        raise ToolValidationError("Path must be non-empty.")
    return VfsPath(_normalize_segments((path,)))


def _build_eval_globals(
    snapshot: VirtualFileSystem, reads: tuple[EvalFileRead, ...]
) -> dict[str, str]:
    values: dict[str, str] = {}
    for read in reads:
        alias = _alias_for_path(read.path)
        file = _require_file(snapshot, read.path)
        values[alias] = file.content
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
    symtable: MutableMapping[str, object]
    node_handlers: MutableMapping[str, object] | None
    error: list[object]

    def eval(self, expression: str) -> object: ...


def _sanitize_interpreter(interpreter: InterpreterProtocol) -> None:
    module = _load_asteval_module()

    for name in getattr(module, "ALL_DISALLOWED", ()):  # pragma: no cover - defensive
        _ = interpreter.symtable.pop(name, None)
    node_handlers = getattr(interpreter, "node_handlers", None)
    if isinstance(node_handlers, MutableMapping):
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


class _AstevalToolSuite:
    def __init__(self, *, section: AstevalSection) -> None:
        super().__init__()
        self._section = section

    def run(
        self, params: EvalParams, *, context: ToolContext
    ) -> ToolResult[EvalResult]:
        ensure_context_uses_session(context=context, session=self._section.session)
        del context
        session = self._section.session
        code = _normalize_code(params.code)
        reads = _normalize_reads(params.reads)
        writes = _normalize_writes(params.writes)
        read_paths = {read.path.segments for read in reads}
        write_paths = {write.path.segments for write in writes}
        if read_paths & write_paths:
            raise ToolValidationError("Reads and writes must not target the same path.")

        snapshot = select_latest(session, VirtualFileSystem) or VirtualFileSystem()
        read_globals = _build_eval_globals(snapshot, reads)
        user_globals = _parse_user_globals(params.globals)

        interpreter = _create_interpreter()
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        write_queue: list[EvalFileWrite] = list(writes)
        helper_writes: list[EvalFileWrite] = []
        write_targets = {write.path.segments for write in write_queue}
        builtin_print = builtins.print
        pending_write_attempted = bool(write_queue)

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
            _ = stdout_buffer.write(text)
            _ = stdout_buffer.write(actual_end)
            if flush:
                _ = stdout_buffer.flush()

        def read_text(path: str) -> str:
            normalized = _normalize_vfs_path(_parse_string_path(path))
            file = _require_file(snapshot, normalized)
            return file.content

        def write_text(path: str, content: str, mode: str = "create") -> None:
            nonlocal pending_write_attempted
            pending_write_attempted = True
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
            if key in write_targets:
                raise ToolValidationError("Duplicate write targets detected.")
            write_targets.add(key)
            helper_writes.append(helper_write)

        symtable = interpreter.symtable
        symtable.update(_merge_globals(read_globals, user_globals))
        symtable["vfs_reads"] = dict(read_globals)
        symtable["read_text"] = read_text
        symtable["write_text"] = write_text
        symtable["print"] = sandbox_print

        all_keys = set(symtable)
        captured_errors: list[str] = []
        value_repr: str | None = None
        stderr_text = ""
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

        param_writes = tuple(write_queue)
        pending_writes = pending_write_attempted or bool(helper_writes)
        if stderr and not value_repr:
            final_writes: tuple[EvalFileWrite, ...] = ()
            if pending_writes:
                message = (
                    "Evaluation failed; pending file writes were discarded. "
                    "Review stderr details in the payload."
                )
            else:
                message = "Evaluation failed; review stderr details in the payload."
        else:
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
            final_writes = tuple(resolved_param_writes + helper_writes)
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

        helper_writes_tuple = tuple(helper_writes)
        pending_sources: Sequence[EvalFileWrite] = (
            final_writes or param_writes + helper_writes_tuple
        )
        summary = _summarize_writes(pending_sources)
        if pending_writes and summary:
            message = f"{message} {summary}"

        globals_payload: dict[str, str] = {}
        visible_keys = {
            key for key in symtable if key not in all_keys and not key.startswith("_")
        }
        visible_keys.update(user_globals.keys())
        for key in visible_keys:
            globals_payload[key] = _format_value(symtable.get(key))
        globals_payload.update(
            {f"vfs:{alias}": content for alias, content in read_globals.items()}
        )

        result = EvalResult(
            value_repr=value_repr,
            stdout=stdout,
            stderr=stderr,
            globals=globals_payload,
            reads=reads,
            writes=final_writes,
        )

        _LOGGER.debug(
            "Asteval evaluation completed.",
            event="asteval_run",
            context={
                "stdout_len": len(stdout),
                "stderr_len": len(stderr),
                "write_count": len(final_writes),
                "code_preview": code[:200],
            },
        )

        return ToolResult(message=message, value=result)


def _make_eval_result_reducer() -> TypedReducer[VirtualFileSystem]:
    def reducer(
        slice_values: tuple[VirtualFileSystem, ...],
        event: ReducerEvent,
        *,
        context: ReducerContextProtocol,
    ) -> tuple[VirtualFileSystem, ...]:
        del context
        previous = slice_values[-1] if slice_values else VirtualFileSystem()
        value = cast(EvalResult, event.value)
        if not value.writes:
            return (previous,)
        snapshot = _apply_writes(previous, value.writes)
        return (snapshot,)

    return reducer


def normalize_eval_code(code: str) -> str:
    return _normalize_code(code)


def normalize_eval_reads(reads: Iterable[EvalFileRead]) -> tuple[EvalFileRead, ...]:
    return _normalize_reads(reads)


def normalize_eval_writes(
    writes: Iterable[EvalFileWrite],
) -> tuple[EvalFileWrite, ...]:
    return _normalize_writes(writes)


def normalize_eval_write(write: EvalFileWrite) -> EvalFileWrite:
    return _normalize_write(write)


def parse_eval_globals(payload: Mapping[str, str]) -> dict[str, object]:
    return _parse_user_globals(payload)


def alias_for_eval_path(path: VfsPath) -> str:
    return _alias_for_path(path)


def summarize_eval_writes(writes: Sequence[EvalFileWrite]) -> str | None:
    return _summarize_writes(writes)


def make_eval_result_reducer() -> TypedReducer[VirtualFileSystem]:
    return _make_eval_result_reducer()


class AstevalSection(MarkdownSection[_AstevalSectionParams]):
    """Prompt section exposing the :mod:`asteval` evaluation tool."""

    def __init__(
        self,
        *,
        session: Session,
        accepts_overrides: bool = False,
    ) -> None:
        self._session = session
        session.register_reducer(
            EvalResult,
            _make_eval_result_reducer(),
            slice_type=VirtualFileSystem,
        )
        tool_suite = _AstevalToolSuite(section=self)
        tool = Tool[EvalParams, EvalResult](
            name="evaluate_python",
            description=(
                "Run a short Python expression or script in a sandbox. Supports "
                "preloading VFS files, staging writes, and returning captured "
                "stdout, stderr, and result data."
            ),
            handler=tool_suite.run,
            accepts_overrides=accepts_overrides,
        )
        super().__init__(
            title="Python Evaluation Tool",
            key="tools.asteval",
            template=_EVAL_TEMPLATE,
            default_params=_AstevalSectionParams(),
            tools=(tool,),
            accepts_overrides=accepts_overrides,
        )

    @property
    def session(self) -> Session:
        return self._session


__all__ = [
    "AstevalSection",
    "EvalFileRead",
    "EvalFileWrite",
    "EvalParams",
    "EvalResult",
    "alias_for_eval_path",
    "make_eval_result_reducer",
    "normalize_eval_code",
    "normalize_eval_reads",
    "normalize_eval_write",
    "normalize_eval_writes",
    "parse_eval_globals",
    "summarize_eval_writes",
]
