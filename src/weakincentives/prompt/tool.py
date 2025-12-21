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

# pyright: reportImportCycles=false

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence as SequenceABC
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    Final,
    Literal,
    Protocol,
    TypeVar,
    cast,
    get_args,
    overload,
)

from ..budget import BudgetTracker
from ..deadlines import Deadline
from ..runtime.snapshotable import Snapshotable
from ..types.dataclass import (
    SupportsDataclass,
    SupportsDataclassOrNone,
    SupportsToolResult,
)
from .errors import PromptValidationError
from .tool_result import ToolResult
from .tool_validation import (
    ParamsType,
    coerce_none_type as _coerce_none_type,
    normalize_result_annotation,
    resolve_annotations,
    resolve_wrapped_description,
    resolve_wrapped_params_type,
    resolve_wrapped_result_annotation,
    validate_description,
    validate_examples,
    validate_handler_if_present,
    validate_name,
    validate_parameter_count,
)

_EXPECTED_TYPE_ARGUMENTS: Final = 2
_NONE_TYPE: Final = type(None)


if TYPE_CHECKING:
    from ..contrib.tools.filesystem import Filesystem
    from ..runtime.session.protocols import SessionProtocol
    from .protocols import (
        PromptProtocol,
        ProviderAdapterProtocol,
        RenderedPromptProtocol,
    )

ParamsT_contra = TypeVar(
    "ParamsT_contra", bound=SupportsDataclassOrNone, contravariant=True
)
ResultT_co = TypeVar("ResultT_co", bound=SupportsToolResult)
ParamsT_runtime = TypeVar("ParamsT_runtime", bound=SupportsDataclassOrNone)
ResultT_runtime = TypeVar("ResultT_runtime", bound=SupportsToolResult)

_ResourceT = TypeVar("_ResourceT")


@dataclass(slots=True, frozen=True)
class ResourceRegistry:
    """Typed container for runtime resources available to tool handlers.

    ResourceRegistry provides type-safe access to runtime services like
    filesystem backends, HTTP clients, or tracers without requiring
    dedicated fields on ToolContext.

    Resources are stored by their type and retrieved via the ``get`` method:

    .. code-block:: python

        fs = context.resources.get(Filesystem)
        if fs is not None:
            content = fs.read("path/to/file")

    This design allows extensibility without bloating the core dataclass.
    Common resources like ``Filesystem`` have sugar properties on
    ``ToolContext`` for convenience.
    """

    _entries: Mapping[type[object], object] = field(
        default_factory=lambda: MappingProxyType({}),
    )

    @overload
    def get(self, resource_type: type[_ResourceT]) -> _ResourceT | None: ...

    @overload
    def get(
        self, resource_type: type[_ResourceT], default: _ResourceT
    ) -> _ResourceT: ...

    def get(
        self, resource_type: type[_ResourceT], default: _ResourceT | None = None
    ) -> _ResourceT | None:
        """Return the resource of the given type, or ``default`` if absent."""
        value = self._entries.get(resource_type)
        if value is None:
            return default
        return cast(_ResourceT, value)

    def __contains__(self, resource_type: type[object]) -> bool:
        """Check if a resource of the given type is registered."""
        return resource_type in self._entries

    def snapshotable_resources(self) -> Mapping[type[object], Snapshotable]:
        """Return all resources that implement Snapshotable.

        Returns:
            Mapping from resource type to snapshotable resource instance.
            Only resources that implement the Snapshotable protocol are included.
        """
        return {k: v for k, v in self._entries.items() if isinstance(v, Snapshotable)}

    @staticmethod
    def build(mapping: Mapping[type[object], object]) -> ResourceRegistry:
        """Construct a registry from a type-to-instance mapping.

        Use protocol types as keys to enable protocol-based lookup:

        .. code-block:: python

            registry = ResourceRegistry.build({
                Filesystem: InMemoryFilesystem(),
                HTTPClient: MyHTTPClient(),
            })

            # Now protocol-based lookup works:
            fs = registry.get(Filesystem)  # Returns the InMemoryFilesystem
        """
        filtered = {k: v for k, v in mapping.items() if v is not None}
        return ResourceRegistry(_entries=MappingProxyType(filtered))


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

    Resources are available through the typed ``resources`` registry:

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
    """

    prompt: PromptProtocol[Any]
    rendered_prompt: RenderedPromptProtocol[Any] | None
    adapter: ProviderAdapterProtocol[Any]
    session: SessionProtocol
    deadline: Deadline | None = None
    resources: ResourceRegistry = field(default_factory=ResourceRegistry)

    @property
    def filesystem(self) -> Filesystem | None:
        """Return the filesystem resource, if available.

        This is sugar for ``self.resources.get(Filesystem)``.
        """
        # Import here to avoid circular import at module load time
        from ..contrib.tools.filesystem import Filesystem

        return self.resources.get(Filesystem)

    @property
    def budget_tracker(self) -> BudgetTracker | None:
        """Return the budget tracker resource, if available.

        This is sugar for ``self.resources.get(BudgetTracker)``.
        """
        return self.resources.get(BudgetTracker)


def _normalize_specialization(item: object) -> tuple[object, object]:
    if not isinstance(item, tuple):
        raise TypeError("Tool[...] expects two type arguments (ParamsT, ResultT).")
    normalized = cast(SequenceABC[object], item)
    if len(normalized) != _EXPECTED_TYPE_ARGUMENTS:
        raise TypeError("Tool[...] expects two type arguments (ParamsT, ResultT).")
    return _coerce_none_type(normalized[0]), _coerce_none_type(normalized[1])


class ToolHandler(Protocol[ParamsT_contra, ResultT_co]):
    """Callable protocol implemented by tool handlers."""

    def __call__(
        self, params: ParamsT_contra, *, context: ToolContext
    ) -> ToolResult[ResultT_co]: ...


@dataclass(slots=True)
class Tool[ParamsT: SupportsDataclassOrNone, ResultT: SupportsToolResult]:
    """Describe a callable tool exposed by prompt sections."""

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
        params_type, raw_result_annotation = self._resolve_type_arguments()

        result_type, result_container = normalize_result_annotation(
            raw_result_annotation,
            params_type,
        )

        self.params_type = cast(type[ParamsT], params_type)
        self.result_type = result_type
        self.result_container = result_container
        self._result_annotation = raw_result_annotation

        self.name = validate_name(self.name, params_type)
        self.description = validate_description(self.description, params_type)
        self.examples = validate_examples(
            self.examples,
            params_type,
            result_type,
            result_container,
        )

        validate_handler_if_present(
            self.handler,
            params_type,
            raw_result_annotation,
        )

    def _resolve_type_arguments(
        self,
    ) -> tuple[ParamsType, ResultT]:
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
                        params_type = cast(ParamsType, _coerce_none_type(params_arg))
                        raw_result_annotation = cast(
                            ResultT, _coerce_none_type(result_arg)
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
        params_candidate, result_candidate = _normalize_specialization(item)
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
    def wrap(
        fn: ToolHandler[ParamsT_runtime, ResultT_runtime],
    ) -> Tool[ParamsT_runtime, ResultT_runtime]:
        """Create a Tool from a handler using its name and docstring."""

        description = resolve_wrapped_description(fn)

        signature = inspect.signature(fn)
        parameter = validate_parameter_count(
            list(signature.parameters.values()),
            _NONE_TYPE,
        )[0]

        hints = resolve_annotations(fn)
        params_type: ParamsType = resolve_wrapped_params_type(parameter, hints)
        normalized_result = cast(
            ResultT_runtime,
            resolve_wrapped_result_annotation(
                signature,
                hints,
                params_type,
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


__all__ = [
    "ResourceRegistry",
    "Tool",
    "ToolContext",
    "ToolExample",
    "ToolHandler",
    "ToolResult",
]
