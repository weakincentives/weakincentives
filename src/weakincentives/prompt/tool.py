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
from collections.abc import Callable, Sequence as SequenceABC
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    ClassVar,
    Final,
    Literal,
    Protocol,
    TypeGuard,
    TypeVar,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from ..budget import BudgetTracker
from ..dataclasses import Constructable, FrozenDataclass, allow_construction
from ..deadlines import Deadline
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
_DESCRIPTION_MAX_LENGTH: Final = 200
_EXPECTED_TYPE_ARGUMENTS: Final = 2
_HANDLER_PARAMETER_COUNT: Final = 2
_VARIADIC_TUPLE_LENGTH: Final = 2
_NONE_TYPE: Final = type(None)

_NAME_PATTERN: Final[re.Pattern[str]] = re.compile(
    rf"^[a-z0-9_-]{{{_NAME_MIN_LENGTH},{_NAME_MAX_LENGTH}}}$"
)


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

type ParamsType = type[SupportsDataclass | None]
type ResultType = type[SupportsDataclass | None]

# Contravariance for ParamsT allows a handler accepting broader param types
# to substitute for one requiring narrower types (Liskov substitution).
ParamsT_contra = TypeVar(
    "ParamsT_contra", bound=SupportsDataclassOrNone, contravariant=True
)
ResultT = TypeVar("ResultT", bound=SupportsToolResult)


def _is_params_type(x: object) -> TypeGuard[ParamsType]:
    """Type guard narrowing *x* to ``ParamsType`` when it is a concrete type."""
    return isinstance(x, type)


def _is_result_type(x: object) -> TypeGuard[ResultType]:
    """Type guard narrowing *x* to ``ResultType`` when it is a concrete type."""
    return isinstance(x, type)


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


def _normalize_specialization(item: object) -> tuple[object, object]:
    if not isinstance(item, tuple):
        raise TypeError("Tool[...] expects two type arguments (ParamsT, ResultT).")
    # isinstance narrows to tuple[Unknown, ...]; cast to provide element types
    normalized = cast("tuple[object, ...]", item)
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


class ToolHandler(Protocol[ParamsT_contra, ResultT]):
    """Callable protocol implemented by tool handlers."""

    def __call__(
        self, params: ParamsT_contra, *, context: ToolContext
    ) -> ToolResult[ResultT]: ...


@FrozenDataclass()
class Tool[ParamsT: SupportsDataclassOrNone, ResultT: SupportsToolResult](
    Constructable
):
    """Describe a callable tool exposed by prompt sections."""

    name: str
    description: str
    handler: ToolHandler[ParamsT, ResultT] | None
    examples: tuple[ToolExample[ParamsT, ResultT], ...] = field(
        default_factory=tuple,
    )
    accepts_overrides: bool = True

    # --- Class-level type resolution (set by __class_getitem__) ---
    # These are NOT instance fields.  They are class attributes computed
    # once per specialization (e.g. Tool[MyParams, MyResult]).

    _specialized_params_type: ClassVar[type | None] = None
    _specialized_result_annotation: ClassVar[object | None] = None
    _specialized_result_type: ClassVar[type | None] = None
    _specialized_result_container: ClassVar[Literal["object", "array"]] = "object"

    @property
    def params_type(self) -> type[ParamsT]:
        """The concrete dataclass type for tool parameters."""
        return cast(type[ParamsT], type(self)._specialized_params_type)

    @property
    def result_type(self) -> type[SupportsDataclass | None]:
        """The concrete dataclass type for tool results."""
        rt = type(self)._specialized_result_type
        if _is_result_type(rt):
            return rt
        return _NONE_TYPE

    @property
    def result_container(self) -> Literal["object", "array"]:
        """Whether the result is a single object or array of objects."""
        return type(self)._specialized_result_container

    @classmethod
    def _require_params_type(cls) -> ParamsType:
        """Return the specialized params type or raise if unspecialized."""
        pt = cls._specialized_params_type
        if _is_params_type(pt):
            return pt
        raise PromptValidationError(
            "Tool must be instantiated with concrete type arguments.",
            placeholder="type_arguments",
        )

    @classmethod
    def create(
        cls,
        *,
        name: str,
        description: str,
        handler: ToolHandler[ParamsT, ResultT] | None,
        examples: tuple[ToolExample[ParamsT, ResultT], ...] = (),
        accepts_overrides: bool = True,
    ) -> Tool[ParamsT, ResultT]:
        """Create a validated Tool instance."""
        params_type = cls._require_params_type()
        raw_result_annotation = cls._specialized_result_annotation
        if raw_result_annotation is None:
            raise PromptValidationError(
                "Tool must be instantiated with concrete type arguments.",
                placeholder="type_arguments",
            )

        # Use pre-computed values from __class_getitem__ if available,
        # otherwise normalize now (deferred from abstract specialization).
        rt = cls._specialized_result_type
        if _is_result_type(rt):
            result_type = rt
            result_container = cls._specialized_result_container
        else:
            result_type, result_container = cls._normalize_result_annotation(
                raw_result_annotation,
                params_type,
            )

        with allow_construction():
            return cls(
                name=cls._validate_name(name, params_type),
                description=cls._validate_description(description, params_type),
                handler=handler,
                examples=cls._validate_examples(
                    examples, params_type, result_type, result_container
                ),
                accepts_overrides=accepts_overrides,
            )

    @staticmethod
    def _validate_name(name: str, params_type: ParamsType) -> str:
        raw_name = name
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

    @staticmethod
    def _validate_description(description: str, params_type: ParamsType) -> str:
        description_clean = description.strip()
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

    @staticmethod
    def _validate_example_output(
        example_output: object,
        params_type: ParamsType,
        result_type: ResultType,
        result_container: Literal["object", "array"],
    ) -> None:
        if result_type is _NONE_TYPE:
            if example_output is not None:
                raise PromptValidationError(
                    "Tool example output must be None when ResultT is None.",
                    dataclass_type=params_type,
                    placeholder="examples",
                )
            return
        if result_container == "array":
            if not isinstance(example_output, SequenceABC) or isinstance(
                example_output, (str, bytes, bytearray)
            ):
                raise PromptValidationError(
                    "Tool example output must be a sequence of ResultT dataclass instances.",
                    dataclass_type=params_type,
                    placeholder="examples",
                )

            # isinstance narrows to Sequence[Unknown]; cast to provide element type
            sequence_output = cast("SequenceABC[object]", example_output)
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

    @staticmethod
    def _validate_examples(
        examples: tuple[ToolExample[ParamsT, ResultT], ...] | tuple[object, ...],
        params_type: ParamsType,
        result_type: ResultType,
        result_container: Literal["object", "array"],
    ) -> tuple[ToolExample[ParamsT, ResultT], ...]:
        if not examples:
            return ()

        normalized_examples: list[ToolExample[ParamsT, ResultT]] = []
        for example in examples:
            if not isinstance(example, ToolExample):
                raise PromptValidationError(
                    "Tool examples must be ToolExample instances.",
                    dataclass_type=params_type,
                    placeholder="examples",
                )

            typed_example = cast(ToolExample[ParamsT, ResultT], example)

            Tool._validate_example_description(typed_example.description, params_type)
            Tool._validate_example_input(typed_example.input, params_type)
            Tool._validate_example_output(
                typed_example.output, params_type, result_type, result_container
            )

            normalized_examples.append(typed_example)

        return tuple(normalized_examples)

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
    def _resolve_annotations(
        callable_handler: Callable[..., ToolResult[ResultT]],
    ) -> dict[str, object]:
        try:
            return get_type_hints(callable_handler, include_extras=True)
        except Exception:
            return {}

    @staticmethod
    def _normalize_result_annotation(
        annotation: object,
        params_type: ParamsType,
    ) -> tuple[ResultType, Literal["object", "array"]]:
        if annotation is None:
            return _NONE_TYPE, "object"
        if _is_result_type(annotation):
            return annotation, "object"

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

        if not _is_result_type(element):
            raise PromptValidationError(
                "Tool ResultT must be a dataclass type or a sequence of dataclasses.",
                dataclass_type=params_type,
                placeholder="ResultT",
            )

        return element, "array"

    @classmethod
    def __class_getitem__(
        cls, item: object
    ) -> type[Tool[SupportsDataclassOrNone, SupportsToolResult]]:
        params_candidate, result_candidate = _normalize_specialization(item)
        if not _is_params_type(params_candidate):
            raise TypeError("Tool ParamsT type argument must be a type.")
        params_type = params_candidate
        result_annotation = result_candidate

        # Try to normalize eagerly.  This succeeds for concrete types
        # (e.g. Tool[MyParams, MyResult]) but can fail for abstract type
        # aliases used in cast() expressions.  When it fails we defer
        # normalization to create().
        try:
            result_type, result_container = cls._normalize_result_annotation(
                result_annotation,
                params_type,
            )
        except PromptValidationError:
            result_type = None
            result_container = "object"

        class _SpecializedTool(cls):  # ty: ignore[shadowed-type-variable]  # dynamic class
            _specialized_params_type = params_type
            _specialized_result_annotation = result_annotation
            _specialized_result_type = result_type
            _specialized_result_container = result_container

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

        coerced = _coerce_none_type(params_annotation)
        if _is_params_type(coerced):
            return coerced
        raise PromptValidationError(
            "Tool handler parameter must be annotated with ParamsT.",
            placeholder="handler",
        )

    @staticmethod
    def _resolve_wrapped_result_annotation(
        signature: inspect.Signature,
        hints: dict[str, object],
        params_type: ParamsType,
    ) -> object:
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

        result_args = get_args(result_annotation)
        try:
            result_arg = next(iter(result_args))
        except StopIteration as error:
            raise PromptValidationError(
                "Tool handler return annotation must be ToolResult[ResultT].",
                dataclass_type=params_type,
                placeholder="return",
            ) from error

        return _coerce_none_type(result_arg)

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
        normalized_result = Tool._resolve_wrapped_result_annotation(
            signature,
            hints,
            params_type,
        )

        tool_type = Tool.__class_getitem__((params_type, normalized_result))
        specialized_tool_type = cast(
            "type[Tool[ParamsT_runtime, ResultT_runtime]]",
            tool_type,
        )

        handler_name = getattr(fn, "__name__", type(fn).__name__)

        return specialized_tool_type.create(
            name=handler_name,
            description=description,
            handler=fn,
        )


__all__ = [
    "Tool",
    "ToolContext",
    "ToolExample",
    "ToolHandler",
    "ToolResult",
]
