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

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Final,
    Literal,
    Protocol,
    TypeVar,
    cast,
    get_args,
)

from ..budget import BudgetTracker
from ..deadlines import Deadline
from ..types.dataclass import (
    SupportsDataclass,
    SupportsDataclassOrNone,
    SupportsToolResult,
)
from .errors import PromptValidationError
from .tool_result import ToolResult
from .tool_validation import (
    ParamsType,
    coerce_none_type,
    get_type_resolver,
    get_validator,
    normalize_specialization,
)

_EXPECTED_TYPE_ARGUMENTS: Final = 2
_NONE_TYPE: Final = type(None)


if TYPE_CHECKING:
    from ..filesystem import Filesystem
    from ..runtime.run_context import RunContext
    from ..runtime.session.protocols import SessionProtocol
    from ..runtime.watchdog import Heartbeat
    from ._prompt_resources import PromptResources
    from .protocols import (
        PromptProtocol,
        ProviderAdapterProtocol,
        RenderedPromptProtocol,
    )

# Contravariance for ParamsT allows a handler accepting broader param types
# to substitute for one requiring narrower types (Liskov substitution).
ParamsT_contra = TypeVar(
    "ParamsT_contra", bound=SupportsDataclassOrNone, contravariant=True
)
ResultT_co = TypeVar("ResultT_co", bound=SupportsToolResult)


@dataclass(slots=True, frozen=True)
class ToolExample[ParamsT: SupportsDataclassOrNone, ResultT: SupportsToolResult]:
    """Representative invocation for a tool documenting inputs and outputs."""

    description: str
    input: ParamsT
    output: ResultT


@dataclass(slots=True, frozen=True)
class ToolContext:
    """Immutable container exposing prompt execution state to handlers.

    ToolContext provides access to prompt metadata, session state, and
    runtime resources during tool execution.

    Resources are available through the ``resources`` context, which
    delegates to the prompt's resource context:

    .. code-block:: python

        fs = context.resources.get(Filesystem)
        tracer = context.resources.get(Tracer)
        budget = context.resources.get(BudgetTracker)

    Common resources have sugar properties for convenience:

    .. code-block:: python

        # These are equivalent:
        context.filesystem
        context.resources.get(Filesystem)

        context.budget_tracker
        context.resources.get(BudgetTracker)

    When running within an AgentLoop or EvalLoop with lease extension enabled,
    tool handlers can call ``context.beat()`` to record a heartbeat during
    long-running operations. This helps extend the message visibility
    timeout and proves that processing is making progress.
    """

    prompt: PromptProtocol[Any]
    rendered_prompt: RenderedPromptProtocol[Any] | None
    adapter: ProviderAdapterProtocol[Any]
    session: SessionProtocol
    deadline: Deadline | None = None
    heartbeat: Heartbeat | None = None
    run_context: RunContext | None = None
    """Execution context with correlation identifiers and metadata."""

    @property
    def resources(self) -> PromptResources:
        """Access resources from the prompt's resource context.

        Returns the active resource context from the prompt. The caller
        must be within ``with prompt.resources:`` for resources to be available.
        """
        return self.prompt.resources

    @property
    def filesystem(self) -> Filesystem | None:
        """Return the filesystem resource, if available.

        This is sugar for ``self.resources.get_optional(Filesystem)``.
        """
        # Import here to avoid circular import at module load time
        from ..filesystem import Filesystem

        return self.resources.get_optional(Filesystem)

    @property
    def budget_tracker(self) -> BudgetTracker | None:
        """Return the budget tracker resource, if available.

        This is sugar for ``self.resources.get_optional(BudgetTracker)``.
        """
        return self.resources.get_optional(BudgetTracker)

    def beat(self) -> None:
        """Record a heartbeat to prove processing is active.

        Tool handlers should call this during long-running operations to
        extend the message visibility timeout. This is a no-op if heartbeat
        is not configured.

        Example::

            def my_long_running_tool(params: Params, *, context: ToolContext) -> ToolResult[None]:
                for chunk in process_chunks(params.data):
                    handle_chunk(chunk)
                    context.beat()  # Prove we're making progress
                return ToolResult.ok(None)
        """
        if self.heartbeat is not None:
            self.heartbeat.beat()


class ToolHandler(Protocol[ParamsT_contra, ResultT_co]):
    """Callable protocol implemented by tool handlers."""

    def __call__(
        self, params: ParamsT_contra, *, context: ToolContext
    ) -> ToolResult[ResultT_co]: ...


@dataclass(slots=True)
class Tool[ParamsT: SupportsDataclassOrNone, ResultT: SupportsToolResult]:
    """Describe a callable tool exposed by prompt sections.

    Tool definitions are validated on construction. Validation is delegated
    to ``ToolValidator`` for reusability and testability.
    """

    name: str
    description: str
    handler: ToolHandler[ParamsT, ResultT] | None
    examples: tuple[ToolExample[ParamsT, ResultT], ...] = field(
        default_factory=tuple,
    )
    params_type: type[ParamsT] = field(init=False, repr=False)
    result_type: type[SupportsDataclass] | type[None] = field(init=False, repr=False)
    result_container: Literal["object", "array"] = field(
        init=False,
        repr=False,
    )
    _result_annotation: ResultT = field(init=False, repr=False)
    accepts_overrides: bool = True

    def __post_init__(self) -> None:
        validator = get_validator()
        resolver = get_type_resolver()

        params_type, raw_result_annotation = self._resolve_type_arguments()

        result_type, result_container = resolver.normalize_result_annotation(
            raw_result_annotation,
            params_type=params_type,
        )

        self.params_type = cast(type[ParamsT], params_type)
        self.result_type = result_type
        self.result_container = result_container
        self._result_annotation = raw_result_annotation

        self.name = validator.validate_name(self.name, params_type=params_type)
        self.description = validator.validate_description(
            self.description, params_type=params_type
        )
        self.examples = cast(
            tuple[ToolExample[ParamsT, ResultT], ...],
            validator.validate_examples(
                self.examples,
                params_type=params_type,
                result_type=result_type,
                result_container=result_container,
            ),
        )

        # Handler validation removed: pyright strict mode catches signature mismatches
        # at development time. Runtime TypeErrors are caught in _handle_tool_exception
        # (tool_executor.py) and converted to ToolResult errors.

    def _resolve_type_arguments(
        self,
    ) -> tuple[ParamsType, ResultT]:
        """Resolve ParamsT and ResultT from class specialization or attributes."""
        params_attr = getattr(self, "params_type", None)
        params_type: ParamsType | None = (
            cast(ParamsType, params_attr) if isinstance(params_attr, type) else None
        )
        raw_result_annotation = getattr(self, "_result_annotation", None)
        if params_type is None or raw_result_annotation is None:
            origin = getattr(self, "__orig_class__", None)
            if origin is not None:  # pragma: no cover - interpreter-specific path
                args = get_args(origin)
                if len(args) == _EXPECTED_TYPE_ARGUMENTS:
                    params_arg, result_arg = args
                    if isinstance(params_arg, type):
                        params_type = cast(ParamsType, coerce_none_type(params_arg))
                        raw_result_annotation = cast(
                            ResultT, coerce_none_type(result_arg)
                        )
        if params_type is None or raw_result_annotation is None:
            raise PromptValidationError(
                "Tool must be instantiated with concrete type arguments.",
                placeholder="type_arguments",
            )

        return params_type, cast(ResultT, raw_result_annotation)

    @classmethod
    def __class_getitem__(
        cls, item: object
    ) -> type[Tool[SupportsDataclassOrNone, SupportsToolResult]]:
        """Support Tool[ParamsT, ResultT] generic specialization."""
        params_candidate, result_candidate = normalize_specialization(item)
        if not isinstance(params_candidate, type):
            raise TypeError("Tool ParamsT type argument must be a type.")
        params_type = cast(ParamsType, params_candidate)
        result_annotation = cast(ResultT, result_candidate)

        class _SpecializedTool(cls):  # ty: ignore[invalid-generic-class]  # dynamic class
            def __post_init__(self) -> None:
                self.params_type = cast(type[ParamsT], params_type)
                self._result_annotation = result_annotation
                super().__post_init__()

        _SpecializedTool.__name__ = cls.__name__
        _SpecializedTool.__qualname__ = cls.__qualname__
        _SpecializedTool.__module__ = cls.__module__
        return cast(
            "type[Tool[SupportsDataclassOrNone, SupportsToolResult]]",
            _SpecializedTool,
        )

    @staticmethod
    def wrap[
        ParamsT_runtime: SupportsDataclassOrNone,
        ResultT_runtime: SupportsToolResult,
    ](
        fn: ToolHandler[ParamsT_runtime, ResultT_runtime],
    ) -> Tool[ParamsT_runtime, ResultT_runtime]:
        """Create a Tool from a handler using its name and docstring.

        The handler function must have:
        - A docstring (used as tool description)
        - Type annotation for the params parameter
        - Return type annotation of ToolResult[ResultT]
        - Exactly two parameters: (params: ParamsT, *, context: ToolContext)

        Example::

            def my_tool(params: MyParams, *, context: ToolContext) -> ToolResult[MyResult]:
                \"\"\"Does something useful.\"\"\"
                return ToolResult.ok(MyResult(...), message="Done")

            tool = Tool.wrap(my_tool)
        """
        validator = get_validator()
        resolver = get_type_resolver()

        description = resolver.resolve_wrapped_description(fn)

        signature = inspect.signature(fn)
        parameter = validator.validate_parameter_count(
            list(signature.parameters.values()),
            params_type=_NONE_TYPE,
        )[0]

        hints = resolver.resolve_annotations(fn)
        params_type: ParamsType = resolver.resolve_wrapped_params_type(parameter, hints)
        normalized_result = cast(
            ResultT_runtime,
            resolver.resolve_wrapped_result_annotation(
                signature,
                hints,
                params_type=params_type,
            ),
        )

        tool_type = Tool.__class_getitem__((params_type, normalized_result))
        specialized_tool_type = cast(
            "type[Tool[ParamsT_runtime, ResultT_runtime]]",
            tool_type,
        )

        handler_name = getattr(fn, "__name__", type(fn).__name__)

        return specialized_tool_type(
            name=handler_name,
            description=description,
            handler=fn,
        )


def create_tool[ParamsT: SupportsDataclassOrNone, ResultT: SupportsToolResult](
    name: str,
    handler: ToolHandler[ParamsT, ResultT],
    *,
    description: str | None = None,
    examples: tuple[ToolExample[ParamsT, ResultT], ...] = (),
    accepts_overrides: bool = True,
) -> Tool[ParamsT, ResultT]:
    """Create and validate a tool definition from a handler.

    This factory function provides a convenient way to create tools with
    automatic type resolution from the handler signature. If no description
    is provided, the handler's docstring is used.

    Args:
        name: Tool name (must match OpenAI function name constraints).
        handler: Tool handler function.
        description: Tool description (defaults to handler docstring).
        examples: Optional tuple of ToolExample instances.
        accepts_overrides: Whether the tool accepts runtime overrides.

    Returns:
        Validated Tool instance.

    Raises:
        PromptValidationError: If validation fails.

    Example::

        def search(params: SearchParams, *, context: ToolContext) -> ToolResult[SearchResult]:
            \"\"\"Search for items matching the query.\"\"\"
            ...

        tool = create_tool("search", search)
        # Or with custom description:
        tool = create_tool("search", search, description="Custom description.")
    """
    resolver = get_type_resolver()

    # Resolve description from docstring if not provided
    if description is None:
        description = resolver.resolve_wrapped_description(handler)

    # Resolve types from handler signature
    signature = inspect.signature(handler)
    validator = get_validator()
    parameter = validator.validate_parameter_count(
        list(signature.parameters.values()),
        params_type=_NONE_TYPE,
    )[0]

    hints = resolver.resolve_annotations(handler)
    params_type: ParamsType = resolver.resolve_wrapped_params_type(parameter, hints)
    normalized_result = cast(
        ResultT,
        resolver.resolve_wrapped_result_annotation(
            signature,
            hints,
            params_type=params_type,
        ),
    )

    # Create specialized tool type
    tool_type = Tool.__class_getitem__((params_type, normalized_result))
    specialized_tool_type = cast(
        "type[Tool[ParamsT, ResultT]]",
        tool_type,
    )

    return specialized_tool_type(
        name=name,
        description=description,
        handler=handler,
        examples=examples,
        accepts_overrides=accepts_overrides,
    )


# Re-export for backward compatibility
_coerce_none_type = coerce_none_type
_normalize_specialization = normalize_specialization


__all__ = [
    "Tool",
    "ToolContext",
    "ToolExample",
    "ToolHandler",
    "ToolResult",
    "create_tool",
]
