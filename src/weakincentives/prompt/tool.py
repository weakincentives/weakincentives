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
import types
from collections.abc import Callable, Sequence as SequenceABC
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Final,
    Literal,
    Protocol,
    TypeVar,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from ..budget import BudgetTracker
from ..deadlines import Deadline
from ..types.dataclass import (
    SupportsDataclass,
    SupportsDataclassOrNone,
    SupportsToolResult,
    is_dataclass_instance,
)
from ._validation import (
    EXAMPLE_DESCRIPTION_VALIDATOR,
    TOOL_DESCRIPTION_VALIDATOR,
    TOOL_NAME_VALIDATOR,
)
from .errors import PromptValidationError
from .tool_result import ToolResult

_EXPECTED_TYPE_ARGUMENTS: Final = 2
_HANDLER_PARAMETER_COUNT: Final = 2
_VARIADIC_TUPLE_LENGTH: Final = 2
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

type ParamsType = type[SupportsDataclass] | type[None]
type ResultType = type[SupportsDataclass] | type[None]

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

        # Handler validation removed: pyright strict mode catches signature mismatches
        # at development time. Runtime TypeErrors are caught in _handle_tool_exception
        # (tool_executor.py) and converted to ToolResult errors.

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
        error = TOOL_NAME_VALIDATOR.validate(self.name)
        if error:
            # Provide context-specific message for name validation
            if "whitespace" in error.message:
                message = "Tool name must not contain surrounding whitespace."
                # Use stripped name as placeholder (original behavior)
                placeholder = self.name.strip()
            elif "pattern" in error.message:
                message = f"Tool name must match the OpenAI function name constraints (pattern: {TOOL_NAME_VALIDATOR.pattern.pattern if TOOL_NAME_VALIDATOR.pattern else ''})."
                placeholder = self.name
            else:
                message = "Tool name must match the OpenAI function name constraints (1-64 lowercase ASCII letters, digits, underscores, or hyphens)."
                placeholder = self.name.strip()
            raise PromptValidationError(
                message,
                dataclass_type=params_type,
                placeholder=placeholder,
            )
        return self.name.strip()

    def _validate_description(self, params_type: ParamsType) -> str:
        error = TOOL_DESCRIPTION_VALIDATOR.validate(self.description)
        if error:
            if "ASCII" in error.message:
                message = "Tool description must be ASCII."
            else:
                message = "Tool description must be 1-200 ASCII characters."
            raise PromptValidationError(
                message,
                dataclass_type=params_type,
                placeholder="description",
            )
        return self.description.strip()

    @staticmethod
    def _validate_example_description(
        description: str, params_type: ParamsType
    ) -> None:
        error = EXAMPLE_DESCRIPTION_VALIDATOR.validate(description)
        if error:
            if "ASCII" in error.message:
                message = "Tool example description must be ASCII."
            else:
                message = "Tool example description must be 1-200 ASCII characters."
            raise PromptValidationError(
                message,
                dataclass_type=params_type,
                placeholder="description",
            )

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
    "Tool",
    "ToolContext",
    "ToolExample",
    "ToolHandler",
    "ToolResult",
]
