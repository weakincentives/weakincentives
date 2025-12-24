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
import re
import types
from collections.abc import Callable, Mapping, Sequence as SequenceABC
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Final,
    Literal,
    Protocol,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from ..budget import BudgetTracker
from ..deadlines import Deadline
from ..runtime.snapshotable import Snapshotable
from ..types.dataclass import (
    SupportsDataclass,
    SupportsDataclassOrNone,
    SupportsToolResult,
    is_dataclass_instance,
)
from .errors import PromptValidationError
from .tool_result import ToolResult

_NAME_MIN_LENGTH: Final = 1
_NAME_MAX_LENGTH: Final = 64
_DESCRIPTION_MIN_LENGTH: Final = 1
_DESCRIPTION_MAX_LENGTH: Final = 200
_EXPECTED_TYPE_ARGUMENTS: Final = 2
_HANDLER_PARAMETER_COUNT: Final = 2
_VARIADIC_TUPLE_LENGTH: Final = 2
_NONE_TYPE: Final = type(None)
_POSITIONAL_PARAMETER_KINDS: Final = {
    inspect.Parameter.POSITIONAL_ONLY,
    inspect.Parameter.POSITIONAL_OR_KEYWORD,
}

_NAME_PATTERN: Final[re.Pattern[str]] = re.compile(
    rf"^[a-z0-9_-]{{{_NAME_MIN_LENGTH},{_NAME_MAX_LENGTH}}}$"
)


if TYPE_CHECKING:
    from ..filesystem import Filesystem
    from ..runtime.session.protocols import SessionProtocol
    from .protocols import (
        PromptProtocol,
        ProviderAdapterProtocol,
        RenderedPromptProtocol,
    )

type ParamsType = type[SupportsDataclass] | type[None]
type ResultType = type[SupportsDataclass] | type[None]


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

    def get[T](self, resource_type: type[T], default: T | None = None) -> T | None:
        """Return the resource of the given type, or ``default`` if absent."""
        value = self._entries.get(resource_type)
        if value is None:
            return default
        return cast(T, value)

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

    def merge(self, other: ResourceRegistry) -> ResourceRegistry:
        """Merge two registries, with ``other`` taking precedence on conflicts.

        This enables layered resource injection where caller-provided resources
        override workspace defaults:

        .. code-block:: python

            workspace = ResourceRegistry.build({Filesystem: InMemoryFilesystem()})
            user = ResourceRegistry.build({HTTPClient: MyClient(), Filesystem: custom_fs})
            merged = workspace.merge(user)  # user's Filesystem wins

        Returns:
            A new registry containing all resources from both registries.
            When both registries contain the same type, ``other``'s value is used.
        """
        merged = dict(self._entries)
        merged.update(other._entries)
        return ResourceRegistry(_entries=MappingProxyType(merged))

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
        from ..filesystem import Filesystem

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


def _coerce_none_type(candidate: object) -> object:
    if candidate is None:
        return _NONE_TYPE

    origin = get_origin(candidate)
    if origin is types.UnionType:
        args = get_args(candidate)
        if args and all(isinstance(arg, type) for arg in args):
            non_none = [arg for arg in args if arg is not _NONE_TYPE]
            if non_none:
                return non_none[0]
            return _NONE_TYPE

    return candidate


class ToolHandler[ParamsT: SupportsDataclassOrNone, ResultT: SupportsToolResult](
    Protocol
):
    """Callable protocol implemented by tool handlers."""

    def __call__(
        self, params: ParamsT, *, context: ToolContext
    ) -> ToolResult[ResultT]: ...


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

        result_type, result_container = self._normalize_result_annotation(
            raw_result_annotation,
            params_type,
        )

        self.params_type = cast(type[ParamsT], params_type)
        self.result_type = result_type
        self.result_container = result_container
        self._result_annotation = raw_result_annotation

        self.name = self._validate_name(params_type)
        self.description = self._validate_description(params_type)
        self.examples = self._validate_examples(
            params_type,
            result_type,
        )

        self._validate_handler_if_present(
            params_type,
            raw_result_annotation,  # ty: ignore[invalid-argument-type]  # ty typevar bounds
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

    def _validate_name(self, params_type: ParamsType) -> str:
        raw_name = self.name
        stripped_name = raw_name.strip()
        if raw_name != stripped_name:
            normalized_name = stripped_name
            raise PromptValidationError(
                "Tool name must not contain surrounding whitespace.",
                dataclass_type=params_type,
                placeholder=normalized_name,
            )

        name_clean = raw_name
        if not name_clean:
            raise PromptValidationError(
                "Tool name must match the OpenAI function name constraints (1-64 lowercase ASCII letters, digits, underscores, or hyphens).",
                dataclass_type=params_type,
                placeholder=stripped_name,
            )
        if len(name_clean) > _NAME_MAX_LENGTH or not _NAME_PATTERN.fullmatch(
            name_clean
        ):
            raise PromptValidationError(
                f"Tool name must match the OpenAI function name constraints (pattern: {_NAME_PATTERN.pattern}).",
                dataclass_type=params_type,
                placeholder=name_clean,
            )

        return name_clean

    def _validate_description(self, params_type: ParamsType) -> str:
        description_clean = self.description.strip()
        if not description_clean or len(description_clean) > _DESCRIPTION_MAX_LENGTH:
            raise PromptValidationError(
                "Tool description must be 1-200 ASCII characters.",
                dataclass_type=params_type,
                placeholder="description",
            )
        try:
            _ = description_clean.encode("ascii")
        except UnicodeEncodeError as error:
            raise PromptValidationError(
                "Tool description must be ASCII.",
                dataclass_type=params_type,
                placeholder="description",
            ) from error
        return description_clean

    @staticmethod
    def _validate_example_description(
        description: str, params_type: ParamsType
    ) -> None:
        description_clean = description.strip()
        if not description_clean or len(description_clean) > _DESCRIPTION_MAX_LENGTH:
            raise PromptValidationError(
                "Tool example description must be 1-200 ASCII characters.",
                dataclass_type=params_type,
                placeholder="description",
            )
        try:
            _ = description_clean.encode("ascii")
        except UnicodeEncodeError as error:
            raise PromptValidationError(
                "Tool example description must be ASCII.",
                dataclass_type=params_type,
                placeholder="description",
            ) from error

    @staticmethod
    def _validate_example_input(
        example_input: object,
        params_type: ParamsType,
    ) -> None:
        if params_type is _NONE_TYPE:
            if example_input is not None:
                raise PromptValidationError(
                    "Tool example input must be None when ParamsT is None.",
                    dataclass_type=params_type,
                    placeholder="examples",
                )
            return
        if not is_dataclass_instance(example_input):
            raise PromptValidationError(
                "Tool example input must be a ParamsT dataclass instance.",
                dataclass_type=params_type,
                placeholder="examples",
            )
        if type(example_input) is not params_type:
            raise PromptValidationError(
                "Tool example input must match the tool params type.",
                dataclass_type=params_type,
                placeholder="examples",
            )

    def _validate_example_output(
        self,
        example_output: object,
        params_type: ParamsType,
        result_type: ResultType,
    ) -> None:
        if result_type is _NONE_TYPE:
            if example_output is not None:
                raise PromptValidationError(
                    "Tool example output must be None when ResultT is None.",
                    dataclass_type=params_type,
                    placeholder="examples",
                )
            return
        if self.result_container == "array":
            if not isinstance(example_output, SequenceABC) or isinstance(
                example_output, (str, bytes, bytearray)
            ):
                raise PromptValidationError(
                    "Tool example output must be a sequence of ResultT dataclass instances.",
                    dataclass_type=params_type,
                    placeholder="examples",
                )

            sequence_output = cast(SequenceABC[object], example_output)
            for item in sequence_output:
                if not is_dataclass_instance(item) or type(item) is not result_type:
                    raise PromptValidationError(
                        "Tool example output must be a sequence of ResultT dataclass instances.",
                        dataclass_type=params_type,
                        placeholder="examples",
                    )
            return

        if (
            not is_dataclass_instance(example_output)
            or type(example_output) is not result_type
        ):
            raise PromptValidationError(
                "Tool example output must be a ResultT dataclass instance.",
                dataclass_type=params_type,
                placeholder="examples",
            )

    def _validate_examples(
        self,
        params_type: ParamsType,
        result_type: ResultType,
    ) -> tuple[ToolExample[ParamsT, ResultT], ...]:
        examples_value = cast(tuple[object, ...], self.examples)
        if not examples_value:
            return ()

        normalized_examples: list[ToolExample[ParamsT, ResultT]] = []
        for example in examples_value:
            if not isinstance(example, ToolExample):
                raise PromptValidationError(
                    "Tool examples must be ToolExample instances.",
                    dataclass_type=params_type,
                    placeholder="examples",
                )

            typed_example = cast(ToolExample[ParamsT, ResultT], example)

            self._validate_example_description(typed_example.description, params_type)
            self._validate_example_input(typed_example.input, params_type)
            self._validate_example_output(
                typed_example.output, params_type, result_type
            )

            normalized_examples.append(typed_example)

        return tuple(normalized_examples)

    def _validate_handler_if_present(
        self,
        params_type: ParamsType,
        raw_result_annotation: object,
    ) -> None:
        handler = self.handler
        if handler is None:
            return
        self._validate_handler(
            handler,
            params_type,
            raw_result_annotation,
        )

    def _validate_handler(
        self,
        handler: object,
        params_type: ParamsType,
        result_annotation: object,
    ) -> None:
        if not callable(handler):
            raise PromptValidationError(
                "Tool handler must be callable.",
                dataclass_type=params_type,
                placeholder="handler",
            )

        callable_handler = cast(Callable[..., ToolResult[ResultT]], handler)
        signature = inspect.signature(callable_handler)
        parameters = list(signature.parameters.values())

        parameter, context_parameter = self._validate_parameter_count(
            parameters,
            params_type,
        )

        self._validate_parameter_kind(parameter, params_type)
        self._validate_context_parameter(context_parameter, params_type)

        hints = self._resolve_annotations(callable_handler)
        annotation = hints.get(parameter.name, parameter.annotation)
        self._validate_parameter_annotation(annotation, params_type)

        context_annotation = hints.get(
            context_parameter.name, context_parameter.annotation
        )
        self._validate_context_annotation(context_annotation, params_type)

        self._validate_return_annotation(
            hints,
            signature,
            result_annotation,
            params_type,
        )

    @staticmethod
    def _validate_parameter_count(
        parameters: list[inspect.Parameter],
        params_type: ParamsType,
    ) -> tuple[inspect.Parameter, inspect.Parameter]:
        if len(parameters) != _HANDLER_PARAMETER_COUNT:
            raise PromptValidationError(
                "Tool handler must accept exactly one positional argument and the keyword-only 'context' parameter.",
                dataclass_type=params_type,
                placeholder="handler",
            )

        parameter = parameters[0]
        context_parameter = parameters[1]
        return parameter, context_parameter

    @staticmethod
    def _validate_parameter_kind(
        parameter: inspect.Parameter,
        params_type: ParamsType,
    ) -> None:
        if parameter.kind not in _POSITIONAL_PARAMETER_KINDS:
            raise PromptValidationError(
                "Tool handler parameter must be positional.",
                dataclass_type=params_type,
                placeholder="handler",
            )

    @staticmethod
    def _validate_context_parameter(
        context_parameter: inspect.Parameter,
        params_type: ParamsType,
    ) -> None:
        if context_parameter.kind is not inspect.Parameter.KEYWORD_ONLY:
            raise PromptValidationError(
                "Tool handler must declare a keyword-only 'context' parameter.",
                dataclass_type=params_type,
                placeholder="handler",
            )
        if context_parameter.default is not inspect.Signature.empty:
            raise PromptValidationError(
                "Tool handler 'context' parameter must not define a default value.",
                dataclass_type=params_type,
                placeholder="handler",
            )
        if context_parameter.name != "context":
            raise PromptValidationError(
                "Tool handler must name the keyword-only context parameter 'context'.",
                dataclass_type=params_type,
                placeholder="handler",
            )

    @staticmethod
    def _resolve_annotations(
        callable_handler: Callable[..., ToolResult[ResultT]],
    ) -> dict[str, object]:
        try:
            return get_type_hints(callable_handler, include_extras=True)
        except Exception:
            return {}

    @staticmethod
    def _validate_parameter_annotation(
        annotation: object,
        params_type: ParamsType,
    ) -> None:
        if annotation is inspect.Parameter.empty:
            raise PromptValidationError(
                "Tool handler parameter must be annotated with ParamsT.",
                dataclass_type=params_type,
                placeholder="handler",
            )
        annotation_to_validate = annotation
        if get_origin(annotation_to_validate) is Annotated:
            annotation_to_validate = get_args(annotation_to_validate)[0]
        literal_origin = get_origin(annotation_to_validate)
        if literal_origin is Literal:
            literal_args = get_args(annotation_to_validate)
            annotation_to_validate = (
                literal_args[0] if literal_args else annotation_to_validate
            )
        if params_type is _NONE_TYPE and annotation_to_validate in {
            None,
            _NONE_TYPE,
        }:
            return
        if annotation_to_validate is not params_type:
            raise PromptValidationError(
                "Tool handler parameter annotation must match ParamsT.",
                dataclass_type=params_type,
                placeholder="handler",
            )

    @staticmethod
    def _validate_context_annotation(
        context_annotation: object,
        params_type: ParamsType,
    ) -> None:
        if context_annotation is inspect.Parameter.empty:
            raise PromptValidationError(
                "Tool handler must annotate the 'context' parameter with ToolContext.",
                dataclass_type=params_type,
                placeholder="handler",
            )
        context_annotation_to_validate = context_annotation
        if get_origin(context_annotation_to_validate) is Annotated:
            context_annotation_to_validate = get_args(context_annotation_to_validate)[0]
        if context_annotation_to_validate is not ToolContext:
            raise PromptValidationError(
                "Tool handler 'context' annotation must be ToolContext.",
                dataclass_type=params_type,
                placeholder="handler",
            )

    def _validate_return_annotation(
        self,
        hints: dict[str, object],
        signature: inspect.Signature,
        result_annotation: object,
        params_type: ParamsType,
    ) -> None:
        return_annotation = hints.get("return", signature.return_annotation)
        if return_annotation is inspect.Signature.empty:
            raise PromptValidationError(
                "Tool handler must annotate its return value with ToolResult[ResultT].",
                dataclass_type=params_type,
                placeholder="return",
            )
        return_annotation_to_validate = return_annotation
        if get_origin(return_annotation_to_validate) is Annotated:
            return_annotation_to_validate = get_args(return_annotation_to_validate)[0]

        origin = get_origin(return_annotation_to_validate)
        if origin is ToolResult:
            result_args_raw = get_args(return_annotation_to_validate)
            if result_args_raw and self._matches_result_annotation(
                result_args_raw[0],
                result_annotation,
            ):
                return
        raise PromptValidationError(
            "Tool handler return annotation must be ToolResult[ResultT].",
            dataclass_type=params_type,
            placeholder="return",
        )

    @staticmethod
    def _normalize_result_annotation(
        annotation: ResultT,
        params_type: ParamsType,
    ) -> tuple[ResultType, Literal["object", "array"]]:
        if annotation is None:
            return _NONE_TYPE, "object"
        if isinstance(annotation, type):
            return cast(ResultType, annotation), "object"

        origin = get_origin(annotation)
        if origin not in {list, tuple, SequenceABC}:
            raise PromptValidationError(
                "Tool ResultT must be a dataclass type or a sequence of dataclasses.",
                dataclass_type=params_type,
                placeholder="ResultT",
            )

        args = get_args(annotation)
        if origin is tuple and (
            len(args) != _VARIADIC_TUPLE_LENGTH or args[1] is not Ellipsis
        ):
            raise PromptValidationError(
                "Variadic Tuple[ResultT, ...] is required for Tool sequence results.",
                dataclass_type=params_type,
                placeholder="ResultT",
            )
        # list[T] and Sequence[T] always have exactly one type arg
        element = args[0]

        if not isinstance(element, type):
            raise PromptValidationError(
                "Tool ResultT must be a dataclass type or a sequence of dataclasses.",
                dataclass_type=params_type,
                placeholder="ResultT",
            )

        return cast(ResultType, element), "array"

    @staticmethod
    def _matches_result_annotation(candidate: object, expected: object) -> bool:
        candidate = _coerce_none_type(candidate)
        expected = _coerce_none_type(expected)
        if candidate is expected:
            return True

        candidate_origin = get_origin(candidate)
        expected_origin = get_origin(expected)

        if candidate_origin is None or expected_origin is None:
            return False

        sequence_origins = {list, tuple, SequenceABC}
        if candidate_origin in sequence_origins and expected_origin in sequence_origins:
            candidate_args = get_args(candidate)
            expected_args = get_args(expected)
            candidate_element = (
                candidate_args[0]
                if candidate_origin is not tuple
                else candidate_args[0]
                if len(candidate_args) == _VARIADIC_TUPLE_LENGTH
                else None
            )
            expected_element = (
                expected_args[0]
                if expected_origin is not tuple
                else expected_args[0]
                if len(expected_args) == _VARIADIC_TUPLE_LENGTH
                else None
            )
            return candidate_element is expected_element

        return False

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
    def _resolve_wrapped_description[
        ParamsT_runtime: SupportsDataclassOrNone,
        ResultT_runtime: SupportsToolResult,
    ](
        fn: ToolHandler[ParamsT_runtime, ResultT_runtime],
    ) -> str:
        description = inspect.getdoc(fn)
        if description is None:
            raise PromptValidationError(
                "Tool handler must define a docstring to use as the description.",
                placeholder="description",
            )
        return description

    @staticmethod
    def _resolve_wrapped_params_type(
        parameter: inspect.Parameter, hints: dict[str, object]
    ) -> ParamsType:
        annotation = hints.get(parameter.name, parameter.annotation)
        if annotation is inspect.Parameter.empty:
            raise PromptValidationError(
                "Tool handler parameter must be annotated with ParamsT.",
                placeholder="handler",
            )

        params_annotation = annotation
        if get_origin(params_annotation) is Annotated:
            params_annotation = get_args(params_annotation)[0]
        if get_origin(params_annotation) is Literal:
            literal_args = get_args(params_annotation)
            params_annotation = literal_args[0] if literal_args else params_annotation

        return cast(ParamsType, _coerce_none_type(params_annotation))

    @staticmethod
    def _resolve_wrapped_result_annotation(
        signature: inspect.Signature,
        hints: dict[str, object],
        params_type: ParamsType,
    ) -> SupportsToolResult:
        return_annotation = hints.get("return", signature.return_annotation)
        if return_annotation is inspect.Signature.empty:
            raise PromptValidationError(
                "Tool handler must annotate its return value with ToolResult[ResultT].",
                dataclass_type=params_type,
                placeholder="return",
            )
        result_annotation = return_annotation
        if get_origin(result_annotation) is Annotated:
            result_annotation = get_args(result_annotation)[0]

        if get_origin(result_annotation) is not ToolResult:
            raise PromptValidationError(
                "Tool handler return annotation must be ToolResult[ResultT].",
                dataclass_type=params_type,
                placeholder="return",
            )

        try:
            result_arg = next(
                iter(cast(tuple[object, ...], get_args(result_annotation)))
            )
        except StopIteration as error:
            raise PromptValidationError(
                "Tool handler return annotation must be ToolResult[ResultT].",
                dataclass_type=params_type,
                placeholder="return",
            ) from error

        return cast(SupportsToolResult, _coerce_none_type(result_arg))

    @staticmethod
    def wrap[
        ParamsT_runtime: SupportsDataclassOrNone,
        ResultT_runtime: SupportsToolResult,
    ](
        fn: ToolHandler[ParamsT_runtime, ResultT_runtime],
    ) -> Tool[ParamsT_runtime, ResultT_runtime]:
        """Create a Tool from a handler using its name and docstring."""

        description = Tool._resolve_wrapped_description(fn)

        signature = inspect.signature(fn)
        parameter = Tool._validate_parameter_count(
            list(signature.parameters.values()),
            _NONE_TYPE,
        )[0]

        hints = Tool._resolve_annotations(fn)
        params_type: ParamsType = Tool._resolve_wrapped_params_type(parameter, hints)
        normalized_result = cast(
            ResultT_runtime,
            Tool._resolve_wrapped_result_annotation(
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
