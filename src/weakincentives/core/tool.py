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

"""Tools: typed handlers attached to prompt sections."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, final, runtime_checkable

from .errors import ToolError

if TYPE_CHECKING:
    from .prompt import Prompt, RenderedPrompt
    from .protocols import Deadline, ResourceProvider
    from .session import Session

__all__ = [
    "Tool",
    "ToolContext",
    "ToolHandler",
    "ToolPolicy",
    "ToolResult",
]

_TOOL_NAME_RE = re.compile(r"^[a-z0-9_-]{1,64}$")
_MAX_DESCRIPTION = 200


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
    """Runtime context passed to a tool handler."""

    session: Session
    prompt: Prompt
    rendered: RenderedPrompt
    deadline: Deadline | None = None
    resources: ResourceProvider | None = None

    def get_resource[T](self, protocol_type: type[T]) -> T:
        """Return a resource that satisfies ``protocol_type``.

        Raises :class:`LookupError` if no provider is wired in or it cannot
        produce one.
        """
        if self.resources is None:
            raise LookupError(
                f"no ResourceProvider wired; cannot resolve {protocol_type!r}"
            )
        return self.resources.get(protocol_type)


class ToolHandler[ParamsT, ResultT](Protocol):
    """Callable signature every tool handler must satisfy."""

    def __call__(
        self, params: ParamsT, context: ToolContext
    ) -> ToolResult[ResultT]: ...


@final
@dataclass(frozen=True, slots=True)
class Tool[ParamsT, ResultT]:
    """A tool: a name, a 1-line description, and a typed handler."""

    name: str
    description: str
    handler: ToolHandler[ParamsT, ResultT]

    def __post_init__(self) -> None:
        if not _TOOL_NAME_RE.match(self.name):
            raise ToolError(f"invalid tool name: {self.name!r}")
        if not 1 <= len(self.description) <= _MAX_DESCRIPTION:
            raise ToolError(f"tool description must be 1-{_MAX_DESCRIPTION} characters")


@runtime_checkable
class ToolPolicy(Protocol):
    """Interceptor invoked before and after every tool call.

    Implementations must raise
    :class:`weakincentives.core.errors.PolicyDeniedError` from
    :meth:`before_invoke` to refuse a call. The spine ships zero concrete
    policies; extras provide them.
    """

    def before_invoke(self, tool: Tool[Any, Any], context: ToolContext) -> None: ...

    def on_result(self, tool: Tool[Any, Any], result: ToolResult[Any]) -> None: ...


# Re-exporting Callable just so static analyzers see a consistent symbol when
# users alias their own handler types.
_HandlerCallable = Callable[[Any, ToolContext], ToolResult[Any]]
