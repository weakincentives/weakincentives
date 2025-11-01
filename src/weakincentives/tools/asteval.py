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

import ast
import contextlib
import io
import json
import logging
import math
import statistics
import sys
import threading
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, cast

from ..prompt.markdown import MarkdownSection
from ..prompt.tool import Tool, ToolResult
from ..session import Session, select_latest
from ..session.session import DataEvent
from .errors import ToolValidationError
from .vfs import VfsFile, VfsPath, VirtualFileSystem

ExpressionMode = Literal["expr", "statements"]

_logger = logging.getLogger(__name__)

_MAX_CODE_LENGTH: Final[int] = 2_000
_MAX_STREAM_LENGTH: Final[int] = 4_096
_MAX_WRITE_LENGTH: Final[int] = 48_000
_MAX_PATH_DEPTH: Final[int] = 16
_MAX_SEGMENT_LENGTH: Final[int] = 80
_ASCII: Final[str] = "ascii"
_TIMEOUT_SECONDS: Final[float] = 5.0

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
    "- Keep code concise (<=2,000 characters) and prefer expression mode unless you need statements.\n"
    "- Pre-load files via `reads`, or call `read_text(path)` inside code to fetch VFS files.\n"
    "- Stage edits with `write_text(path, content, mode)` or declare them in `writes`. Content must be ASCII.\n"
    "- Globals accept JSON-encoded strings and are parsed before execution.\n"
    "- Execution stops after five seconds; design code to finish quickly."
)


@dataclass(slots=True, frozen=True)
class EvalFileRead:
    path: VfsPath


@dataclass(slots=True, frozen=True)
class EvalFileWrite:
    path: VfsPath
    content: str
    mode: Literal["create", "overwrite", "append"] = "create"


@dataclass(slots=True, frozen=True)
class EvalParams:
    code: str
    mode: ExpressionMode = "expr"
    globals: dict[str, str] = field(default_factory=dict)
    reads: tuple[EvalFileRead, ...] = field(default_factory=tuple)
    writes: tuple[EvalFileWrite, ...] = field(default_factory=tuple)


@dataclass(slots=True, frozen=True)
class EvalResult:
    value_repr: str | None
    stdout: str
    stderr: str
    globals: dict[str, str]
    reads: tuple[EvalFileRead, ...]
    writes: tuple[EvalFileWrite, ...]


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
        value.encode(_ASCII)
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
            files.pop(existing_index)
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


if TYPE_CHECKING:
    from asteval import Interpreter


def _sanitize_interpreter(interpreter: Interpreter) -> None:
    try:
        import asteval  # type: ignore
    except ModuleNotFoundError as error:  # pragma: no cover - configuration guard
        raise RuntimeError("asteval dependency is not installed.") from error

    for name in getattr(asteval, "ALL_DISALLOWED", ()):  # pragma: no cover - defensive
        interpreter.symtable.pop(name, None)
    node_handlers = getattr(interpreter, "node_handlers", None)
    if isinstance(node_handlers, dict):
        for key in ("Eval", "Exec", "Import", "ImportFrom"):
            node_handlers.pop(key, None)


def _create_interpreter() -> Interpreter:
    try:
        from asteval import Interpreter  # type: ignore
    except ModuleNotFoundError as error:  # pragma: no cover - configuration guard
        raise RuntimeError("asteval dependency is not installed.") from error

    interpreter = Interpreter(use_numpy=False, minimal=True)
    interpreter.symtable = dict(_SAFE_GLOBALS)
    _sanitize_interpreter(interpreter)
    return interpreter


def _execute_with_timeout(
    func: Callable[[], object],
) -> tuple[bool, object | None, str]:
    if sys.platform != "win32":  # pragma: no branch - platform check
        import signal

        timed_out = False

        def handler(signum: int, frame: object | None) -> None:  # noqa: ARG001
            nonlocal timed_out
            timed_out = True
            raise TimeoutError

        previous = signal.signal(signal.SIGALRM, handler)
        signal.setitimer(signal.ITIMER_REAL, _TIMEOUT_SECONDS)
        try:
            value = func()
            return False, value, ""
        except TimeoutError:
            return True, None, "Execution timed out."
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
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
    completed.wait(_TIMEOUT_SECONDS)
    if not completed.is_set():
        return True, None, timeout_message
    if "error" in result_container:
        error = cast(Exception, result_container["error"])
        raise error
    return False, result_container.get("value"), error_container["message"]


class _AstevalToolSuite:
    def __init__(self, *, session: Session) -> None:
        self._session = session

    def run(self, params: EvalParams) -> ToolResult[EvalResult]:
        code = _normalize_code(params.code)
        mode = params.mode
        if mode not in {"expr", "statements"}:
            raise ToolValidationError("Unsupported evaluation mode.")
        reads = _normalize_reads(params.reads)
        writes = _normalize_writes(params.writes)
        read_paths = {read.path.segments for read in reads}
        write_paths = {write.path.segments for write in writes}
        if read_paths & write_paths:
            raise ToolValidationError("Reads and writes must not target the same path.")

        snapshot = (
            select_latest(self._session, VirtualFileSystem) or VirtualFileSystem()
        )
        read_globals = _build_eval_globals(snapshot, reads)
        user_globals = _parse_user_globals(params.globals)

        interpreter = _create_interpreter()
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        write_queue: list[EvalFileWrite] = list(writes)
        helper_writes: list[EvalFileWrite] = []
        write_targets = {write.path.segments for write in write_queue}

        if mode == "expr":
            try:
                ast.parse(code, mode="eval")
            except SyntaxError as error:
                raise ToolValidationError(
                    "Expression mode requires a single expression."
                ) from error

        def read_text(path: str) -> str:
            normalized = _normalize_vfs_path(_parse_string_path(path))
            file = _require_file(snapshot, normalized)
            return file.content

        def write_text(path: str, content: str, mode: str = "create") -> None:
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
        symtable.update(user_globals)
        symtable["vfs_reads"] = dict(read_globals)
        symtable["read_text"] = read_text
        symtable["write_text"] = write_text

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
        except ToolValidationError:
            raise
        except Exception as error:  # pragma: no cover - runtime exception
            captured_errors.append(str(error))
        stdout = _truncate_stream(stdout_buffer.getvalue())
        stderr_raw = (
            stderr_text or "\n".join(captured_errors) or stderr_buffer.getvalue()
        )
        stderr = _truncate_stream(stderr_raw)

        param_writes = tuple(write_queue)
        if stderr and not value_repr:
            message = "Evaluation failed. See stderr for details."
            final_writes: tuple[EvalFileWrite, ...] = ()
        else:
            message = "Evaluation completed successfully."
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
                    raise ToolValidationError("Duplicate write targets detected.")
                seen_targets.add(key)

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

        _logger.debug(
            "asteval.run",
            extra={
                "event": "asteval.run",
                "mode": mode,
                "stdout_len": len(stdout),
                "stderr_len": len(stderr),
                "write_count": len(final_writes),
                "code_preview": code[:200],
            },
        )

        return ToolResult(message=message, value=result)


def _make_eval_result_reducer() -> Callable[
    [tuple[VirtualFileSystem, ...], DataEvent], tuple[VirtualFileSystem, ...]
]:
    def reducer(
        slice_values: tuple[VirtualFileSystem, ...], event: DataEvent
    ) -> tuple[VirtualFileSystem, ...]:
        previous = slice_values[-1] if slice_values else VirtualFileSystem()
        value = cast(EvalResult, event.value)
        if not value.writes:
            return (previous,)
        snapshot = _apply_writes(previous, value.writes)
        return (snapshot,)

    return reducer


class AstevalSection(MarkdownSection[_AstevalSectionParams]):
    """Prompt section exposing the :mod:`asteval` evaluation tool."""

    def __init__(self, *, session: Session) -> None:
        self._session = session
        session.register_reducer(
            EvalResult, _make_eval_result_reducer(), slice_type=VirtualFileSystem
        )
        tool_suite = _AstevalToolSuite(session=session)
        tool = Tool[EvalParams, EvalResult](
            name="evaluate_python",
            description="Evaluate a short Python expression in a sandboxed environment with optional VFS access.",
            handler=tool_suite.run,
        )
        super().__init__(
            title="Python Evaluation Tool",
            key="tools.asteval",
            template=_EVAL_TEMPLATE,
            default_params=_AstevalSectionParams(),
            tools=(tool,),
        )


__all__ = [
    "AstevalSection",
    "EvalFileRead",
    "EvalFileWrite",
    "EvalParams",
    "EvalResult",
]
