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

"""Distilled WINK: the novel core in one file.

This module collects the genuinely novel ideas of weakincentives into a
single, dependency-free file modeled after Bottle's all-in-one philosophy.

It is a teaching artifact and a design sandbox, not a drop-in replacement
for the real package. See ``specs/DISTILLED.md`` for scope.

Four ideas, in order of dependency:

1. **Sections + Prompt** - hierarchical markdown that bundles tools.
2. **Tools** - typed handlers attached to sections.
3. **Sessions** - event-sourced state with pure reducers.
4. **Transactions** - snapshot/restore for atomic tool execution.
"""

# Allow ``SliceAccessor`` to reach into ``Session``'s underscore-prefixed
# attributes; the two are intentionally co-designed in this single file.
# pyright: reportPrivateUsage=false

from __future__ import annotations

import re
import textwrap
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, is_dataclass
from string import Template
from typing import Any, cast, final

__all__ = [
    "Append",
    "Prompt",
    "PromptError",
    "RenderedPrompt",
    "Replace",
    "Section",
    "Session",
    "SliceAccessor",
    "SliceOp",
    "Snapshot",
    "Tool",
    "ToolContext",
    "ToolError",
    "ToolResult",
    "WinkError",
    "execute_tool",
    "reducer",
    "tool_transaction",
]


# =============================================================================
# Errors
# =============================================================================


class WinkError(Exception):
    """Base class for all distilled errors."""


class PromptError(WinkError):
    """Raised on prompt validation or render failures."""


class ToolError(WinkError):
    """Raised on tool registration or invocation failures."""


# =============================================================================
# Identifier validation
# =============================================================================

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_TOOL_NAME_RE = re.compile(r"^[a-z0-9_-]{1,64}$")


def _check_name(value: str, *, kind: str) -> None:
    if not _NAME_RE.match(value):
        raise PromptError(f"invalid {kind}: {value!r}")


# =============================================================================
# Tools
# =============================================================================


_MAX_TOOL_DESCRIPTION = 200


@final
@dataclass(frozen=True, slots=True)
class ToolResult[ResultT]:
    """Outcome of a tool invocation."""

    success: bool
    message: str
    value: ResultT | None = None

    @classmethod
    def ok(cls, value: ResultT, message: str = "") -> ToolResult[ResultT]:
        return cls(success=True, message=message, value=value)

    @classmethod
    def error(cls, message: str) -> ToolResult[ResultT]:
        return cls(success=False, message=message, value=None)


@final
@dataclass(frozen=True, slots=True)
class ToolContext:
    """Runtime context handed to a tool handler."""

    session: Session
    prompt: Prompt
    rendered: RenderedPrompt


ToolHandler = Callable[[Any, ToolContext], ToolResult[Any]]


@final
@dataclass(frozen=True, slots=True)
class Tool:
    """A tool: a name, a 1-line description, and a handler."""

    name: str
    description: str
    handler: ToolHandler

    def __post_init__(self) -> None:
        if not _TOOL_NAME_RE.match(self.name):
            raise ToolError(f"invalid tool name: {self.name!r}")
        if not 1 <= len(self.description) <= _MAX_TOOL_DESCRIPTION:
            raise ToolError(
                f"tool description must be 1-{_MAX_TOOL_DESCRIPTION} characters"
            )


# =============================================================================
# Sections + Prompt
# =============================================================================


def _empty_params() -> Mapping[str, Mapping[str, object]]:
    return {}


@final
@dataclass(frozen=True, slots=True)
class Section:
    """A node in a prompt's content tree.

    Sections render as markdown headings with deterministic numbering and
    can attach tools that the prompt collects on render.
    """

    title: str
    key: str
    template: str = ""
    children: tuple[Section, ...] = ()
    tools: tuple[Tool, ...] = ()

    def __post_init__(self) -> None:
        _check_name(self.key, kind="section key")

    def reachable_tools(self) -> tuple[Tool, ...]:
        out: list[Tool] = list(self.tools)
        for child in self.children:
            out.extend(child.reachable_tools())
        return tuple(out)


@final
@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    """Result of rendering a :class:`Prompt`."""

    text: str
    tools: tuple[Tool, ...]


@final
@dataclass(frozen=True, slots=True)
class Prompt:
    """A composable, namespaced prompt.

    ``params`` maps a section key to a dict of substitutions for that
    section's template.
    """

    ns: str
    key: str
    sections: tuple[Section, ...] = ()
    params: Mapping[str, Mapping[str, object]] = field(default_factory=_empty_params)

    def __post_init__(self) -> None:
        _check_name(self.ns, kind="namespace")
        _check_name(self.key, kind="key")
        seen: set[str] = set()
        for section in self.sections:
            for tool in section.reachable_tools():
                if tool.name in seen:
                    raise PromptError(f"duplicate tool name: {tool.name}")
                seen.add(tool.name)

    def render(self) -> RenderedPrompt:
        chunks: list[str] = []
        tools: list[Tool] = []
        for index, section in enumerate(self.sections, start=1):
            self._render_section(section, depth=0, number=str(index), out=chunks)
            tools.extend(section.reachable_tools())
        return RenderedPrompt(text="\n\n".join(chunks), tools=tuple(tools))

    def _render_section(
        self,
        section: Section,
        *,
        depth: int,
        number: str,
        out: list[str],
    ) -> None:
        heading = f"{'#' * (depth + 2)} {number}. {section.title}"
        body = self._render_body(section)
        out.append(f"{heading}\n\n{body}" if body else heading)
        for child_index, child in enumerate(section.children, start=1):
            self._render_section(
                child,
                depth=depth + 1,
                number=f"{number}.{child_index}",
                out=out,
            )

    def _render_body(self, section: Section) -> str:
        if not section.template:
            return ""
        template = Template(textwrap.dedent(section.template).strip())
        substitutions = dict(self.params.get(section.key, {}))
        try:
            return template.substitute(substitutions).strip()
        except KeyError as error:
            placeholder = error.args[0]
            raise PromptError(
                f"missing placeholder {placeholder!r} in section {section.key!r}"
            ) from error


# =============================================================================
# Slice operations and reducers
# =============================================================================


@final
@dataclass(frozen=True, slots=True)
class Replace[SliceT]:
    """Reducer return value: replace the entire slice."""

    items: tuple[SliceT, ...]


@final
@dataclass(frozen=True, slots=True)
class Append[SliceT]:
    """Reducer return value: append items to the slice."""

    items: tuple[SliceT, ...]


SliceOp = Replace[Any] | Append[Any]
TypedReducer = Callable[[tuple[Any, ...], Any], SliceOp]

_REDUCER_ATTR = "__wink_reducer_event__"


def reducer(*, on: type[object]) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Mark a method on a state slice dataclass as a reducer.

    The decorated method must accept ``(self, event)`` and return a
    :class:`SliceOp`. Register it with :meth:`Session.install`.
    """

    def decorate(method: Callable[..., Any]) -> Callable[..., Any]:
        setattr(method, _REDUCER_ATTR, on)
        return method

    return decorate


# =============================================================================
# Sessions
# =============================================================================


@final
@dataclass(frozen=True, slots=True)
class Snapshot:
    """Immutable capture of a session's slice contents."""

    slices: Mapping[type[object], tuple[object, ...]]


class SliceAccessor[SliceT]:
    """Fluent API exposed by ``session[SliceType]``."""

    def __init__(self, session: Session, slice_type: type[SliceT]) -> None:
        super().__init__()
        self._session = session
        self._slice_type = slice_type

    def all(self) -> tuple[SliceT, ...]:
        return cast(
            "tuple[SliceT, ...]",
            self._session._slices.get(self._slice_type, ()),
        )

    def latest(self) -> SliceT | None:
        items = self.all()
        return items[-1] if items else None

    def where(self, predicate: Callable[[SliceT], bool]) -> tuple[SliceT, ...]:
        return tuple(item for item in self.all() if predicate(item))

    def seed(self, value: SliceT) -> None:
        self._session._set_slice(self._slice_type, (value,))

    def append(self, value: SliceT) -> None:
        current = self.all()
        self._session._set_slice(self._slice_type, (*current, value))

    def clear(self) -> None:
        self._session._set_slice(self._slice_type, ())


class Session:
    """Event-sourced state container.

    Slices are typed tuples keyed by class. Mutations flow through pure
    reducers registered against event types.
    """

    def __init__(self) -> None:
        super().__init__()
        self._slices: dict[type[object], tuple[object, ...]] = {}
        self._reducers: dict[type[object], list[_Registration]] = {}
        self._initials: dict[type[object], Callable[[], object]] = {}

    def __getitem__[SliceT](self, slice_type: type[SliceT]) -> SliceAccessor[SliceT]:
        return SliceAccessor(self, slice_type)

    def register[EventT](
        self,
        event_type: type[EventT],
        reducer_fn: TypedReducer,
        *,
        slice_type: type[object],
    ) -> None:
        """Register a free-function reducer for ``event_type``."""
        self._reducers.setdefault(event_type, []).append(
            _Registration(reducer_fn=reducer_fn, slice_type=slice_type)
        )

    def install[SliceT](
        self,
        slice_type: type[SliceT],
        *,
        initial: Callable[[], SliceT] | None = None,
    ) -> None:
        """Register every ``@reducer``-decorated method on a state slice class."""
        _require_frozen_dataclass(slice_type)
        if initial is not None:
            self._initials[slice_type] = cast("Callable[[], object]", initial)
        registrations = _collect_reducer_methods(slice_type)
        if not registrations:
            raise WinkError(f"{slice_type.__name__} declares no @reducer methods")
        for event_type, method_name in registrations:
            self.register(
                event_type,
                _make_method_reducer(slice_type, method_name),
                slice_type=slice_type,
            )

    def dispatch(self, event: object) -> None:
        """Route ``event`` to every reducer registered for ``type(event)``."""
        registrations = self._reducers.get(type(event), [])
        for registration in registrations:
            current = self._slices.get(registration.slice_type, ())
            if not current and registration.slice_type in self._initials:
                current = (self._initials[registration.slice_type](),)
            op = registration.reducer_fn(current, event)
            if isinstance(op, Replace):
                self._slices[registration.slice_type] = tuple(op.items)
            else:
                self._slices[registration.slice_type] = (
                    *current,
                    *op.items,
                )

    def snapshot(self) -> Snapshot:
        return Snapshot(slices=dict(self._slices))

    def restore(self, snapshot: Snapshot) -> None:
        self._slices = dict(snapshot.slices)

    def _set_slice(self, slice_type: type[object], items: tuple[object, ...]) -> None:
        if items:
            self._slices[slice_type] = items
        else:
            _ = self._slices.pop(slice_type, None)


@dataclass(frozen=True, slots=True)
class _Registration:
    reducer_fn: TypedReducer
    slice_type: type[object]


def _require_frozen_dataclass(slice_type: type[object]) -> None:
    if not is_dataclass(slice_type):
        raise WinkError(f"{slice_type.__name__} must be a frozen dataclass")
    params = getattr(slice_type, "__dataclass_params__", None)
    if params is None or not getattr(params, "frozen", False):
        raise WinkError(f"{slice_type.__name__} must be a frozen dataclass")


def _collect_reducer_methods(
    slice_type: type[object],
) -> tuple[tuple[type[object], str], ...]:
    seen: dict[type[object], str] = {}
    for name in dir(slice_type):
        if name.startswith("_"):
            continue
        event_type = getattr(getattr(slice_type, name, None), _REDUCER_ATTR, None)
        if event_type is None:
            continue
        if event_type in seen:
            msg = (
                f"{slice_type.__name__} has duplicate reducers for "
                + event_type.__name__
            )
            raise WinkError(msg)
        seen[event_type] = name
    return tuple(seen.items())


def _make_method_reducer(slice_type: type[object], method_name: str) -> TypedReducer:
    def method_reducer(current: tuple[Any, ...], event: object) -> SliceOp:
        if not current:
            raise WinkError(
                f"no state for {slice_type.__name__}; pass initial= to install()"
            )
        instance = current[-1]
        method = getattr(instance, method_name)
        return cast(SliceOp, method(event))

    method_reducer.__name__ = f"{slice_type.__name__}.{method_name}"
    method_reducer.__qualname__ = method_reducer.__name__
    return method_reducer


# =============================================================================
# Transactions and tool execution
# =============================================================================


@contextmanager
def tool_transaction(session: Session) -> Iterator[Snapshot]:
    """Snapshot before the block; restore on any exception."""
    snapshot = session.snapshot()
    try:
        yield snapshot
    except BaseException:
        session.restore(snapshot)
        raise


def execute_tool(
    prompt: Prompt,
    session: Session,
    name: str,
    params: object,
) -> ToolResult[Any]:
    """Run a tool by name, rolling back session state on any failure.

    Failure is either a raised exception or a ``ToolResult`` with
    ``success=False``. Exceptions other than :class:`WinkError` and
    :class:`AssertionError` are caught and converted into error results.
    """
    rendered = prompt.render()
    matches = [tool for tool in rendered.tools if tool.name == name]
    if not matches:
        raise ToolError(f"no tool named {name!r}")
    tool = matches[0]
    context = ToolContext(session=session, prompt=prompt, rendered=rendered)
    snapshot = session.snapshot()
    try:
        result = tool.handler(params, context)
    except WinkError:
        session.restore(snapshot)
        raise
    except Exception as error:
        session.restore(snapshot)
        return cast(
            "ToolResult[Any]",
            ToolResult.error(f"{type(error).__name__}: {error}"),
        )
    if not result.success:
        session.restore(snapshot)
    return result
