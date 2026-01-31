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
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated

import pytest

from weakincentives.prompt import (
    PromptValidationError,
    ToolContext,
    ToolExample,
    ToolResult,
    ToolValidator,
    TypeResolver,
    create_tool,
)
from weakincentives.prompt.tool_validation import (
    coerce_none_type,
    normalize_specialization,
)


@dataclass
class SampleParams:
    query: str


@dataclass
class SampleResult:
    value: str


@dataclass
class OtherParams:
    count: int


# =============================================================================
# ToolValidator tests
# =============================================================================


class TestValidateName:
    def test_valid_name_passes(self) -> None:
        validator = ToolValidator()
        result = validator.validate_name("my_tool", params_type=SampleParams)
        assert result == "my_tool"

    def test_name_with_hyphens_passes(self) -> None:
        validator = ToolValidator()
        result = validator.validate_name("my-tool", params_type=SampleParams)
        assert result == "my-tool"

    def test_name_with_digits_passes(self) -> None:
        validator = ToolValidator()
        result = validator.validate_name("tool123", params_type=SampleParams)
        assert result == "tool123"

    def test_empty_name_fails(self) -> None:
        validator = ToolValidator()
        with pytest.raises(PromptValidationError, match="must match"):
            validator.validate_name("", params_type=SampleParams)

    def test_name_with_whitespace_fails(self) -> None:
        validator = ToolValidator()
        with pytest.raises(PromptValidationError, match="whitespace"):
            validator.validate_name(" my_tool ", params_type=SampleParams)

    def test_name_with_uppercase_fails(self) -> None:
        validator = ToolValidator()
        with pytest.raises(PromptValidationError, match="must match"):
            validator.validate_name("MyTool", params_type=SampleParams)

    def test_name_too_long_fails(self) -> None:
        validator = ToolValidator()
        long_name = "a" * 65
        with pytest.raises(PromptValidationError, match="must match"):
            validator.validate_name(long_name, params_type=SampleParams)

    def test_name_at_max_length_passes(self) -> None:
        validator = ToolValidator()
        max_name = "a" * 64
        result = validator.validate_name(max_name, params_type=SampleParams)
        assert result == max_name


class TestValidateDescription:
    def test_valid_description_passes(self) -> None:
        validator = ToolValidator()
        result = validator.validate_description(
            "Searches for items.", params_type=SampleParams
        )
        assert result == "Searches for items."

    def test_description_stripped(self) -> None:
        validator = ToolValidator()
        result = validator.validate_description(
            "  Searches for items.  ", params_type=SampleParams
        )
        assert result == "Searches for items."

    def test_empty_description_fails(self) -> None:
        validator = ToolValidator()
        with pytest.raises(PromptValidationError, match="1-200 ASCII"):
            validator.validate_description("", params_type=SampleParams)

    def test_whitespace_only_description_fails(self) -> None:
        validator = ToolValidator()
        with pytest.raises(PromptValidationError, match="1-200 ASCII"):
            validator.validate_description("   ", params_type=SampleParams)

    def test_description_too_long_fails(self) -> None:
        validator = ToolValidator()
        long_desc = "a" * 201
        with pytest.raises(PromptValidationError, match="1-200 ASCII"):
            validator.validate_description(long_desc, params_type=SampleParams)

    def test_description_at_max_length_passes(self) -> None:
        validator = ToolValidator()
        max_desc = "a" * 200
        result = validator.validate_description(max_desc, params_type=SampleParams)
        assert result == max_desc

    def test_non_ascii_description_fails(self) -> None:
        validator = ToolValidator()
        with pytest.raises(PromptValidationError, match="ASCII"):
            validator.validate_description(
                "Search for caf\u00e9 items.", params_type=SampleParams
            )


class TestValidateExampleDescription:
    def test_valid_example_description_passes(self) -> None:
        validator = ToolValidator()
        validator.validate_example_description(
            "Simple lookup", params_type=SampleParams
        )

    def test_empty_example_description_fails(self) -> None:
        validator = ToolValidator()
        with pytest.raises(PromptValidationError, match="example description"):
            validator.validate_example_description("", params_type=SampleParams)

    def test_non_ascii_example_description_fails(self) -> None:
        validator = ToolValidator()
        with pytest.raises(PromptValidationError, match="example description"):
            validator.validate_example_description(
                "test with \u00e9 accent", params_type=SampleParams
            )


class TestValidateExampleInput:
    def test_valid_input_passes(self) -> None:
        validator = ToolValidator()
        validator.validate_example_input(
            SampleParams(query="test"), params_type=SampleParams
        )

    def test_none_input_with_none_params_passes(self) -> None:
        validator = ToolValidator()
        validator.validate_example_input(None, params_type=type(None))

    def test_non_none_input_with_none_params_fails(self) -> None:
        validator = ToolValidator()
        with pytest.raises(PromptValidationError, match="must be None"):
            validator.validate_example_input(
                SampleParams(query="test"), params_type=type(None)
            )

    def test_non_dataclass_input_fails(self) -> None:
        validator = ToolValidator()
        with pytest.raises(PromptValidationError, match="dataclass instance"):
            validator.validate_example_input(
                {"query": "test"}, params_type=SampleParams
            )

    def test_wrong_type_input_fails(self) -> None:
        validator = ToolValidator()
        with pytest.raises(PromptValidationError, match="must match"):
            validator.validate_example_input(
                OtherParams(count=1), params_type=SampleParams
            )


class TestValidateExampleOutput:
    def test_valid_output_passes(self) -> None:
        validator = ToolValidator()
        validator.validate_example_output(
            SampleResult(value="result"),
            params_type=SampleParams,
            result_type=SampleResult,
            result_container="object",
        )

    def test_none_output_with_none_result_passes(self) -> None:
        validator = ToolValidator()
        validator.validate_example_output(
            None,
            params_type=SampleParams,
            result_type=type(None),
            result_container="object",
        )

    def test_non_none_output_with_none_result_fails(self) -> None:
        validator = ToolValidator()
        with pytest.raises(PromptValidationError, match="must be None"):
            validator.validate_example_output(
                SampleResult(value="result"),
                params_type=SampleParams,
                result_type=type(None),
                result_container="object",
            )

    def test_array_output_passes(self) -> None:
        validator = ToolValidator()
        validator.validate_example_output(
            [SampleResult(value="a"), SampleResult(value="b")],
            params_type=SampleParams,
            result_type=SampleResult,
            result_container="array",
        )

    def test_non_sequence_array_output_fails(self) -> None:
        validator = ToolValidator()
        with pytest.raises(PromptValidationError, match="sequence"):
            validator.validate_example_output(
                SampleResult(value="result"),
                params_type=SampleParams,
                result_type=SampleResult,
                result_container="array",
            )

    def test_string_as_array_output_fails(self) -> None:
        validator = ToolValidator()
        with pytest.raises(PromptValidationError, match="sequence"):
            validator.validate_example_output(
                "not a list",
                params_type=SampleParams,
                result_type=SampleResult,
                result_container="array",
            )

    def test_wrong_type_in_array_fails(self) -> None:
        validator = ToolValidator()
        with pytest.raises(PromptValidationError, match="sequence"):
            validator.validate_example_output(
                [OtherParams(count=1)],
                params_type=SampleParams,
                result_type=SampleResult,
                result_container="array",
            )

    def test_non_dataclass_output_fails(self) -> None:
        validator = ToolValidator()
        with pytest.raises(PromptValidationError, match="dataclass instance"):
            validator.validate_example_output(
                {"value": "result"},
                params_type=SampleParams,
                result_type=SampleResult,
                result_container="object",
            )


class TestValidateExamples:
    def test_empty_examples_passes(self) -> None:
        validator = ToolValidator()
        result = validator.validate_examples(
            (),
            params_type=SampleParams,
            result_type=SampleResult,
            result_container="object",
        )
        assert result == ()

    def test_valid_examples_pass(self) -> None:
        validator = ToolValidator()
        examples: tuple[ToolExample[SampleParams, SampleResult], ...] = (
            ToolExample[SampleParams, SampleResult](
                description="simple lookup",
                input=SampleParams(query="test"),
                output=SampleResult(value="result"),
            ),
        )
        result = validator.validate_examples(
            examples,
            params_type=SampleParams,
            result_type=SampleResult,
            result_container="object",
        )
        assert result == examples

    def test_non_tool_example_fails(self) -> None:
        validator = ToolValidator()
        with pytest.raises(PromptValidationError, match="ToolExample instances"):
            validator.validate_examples(
                ("not an example",),
                params_type=SampleParams,
                result_type=SampleResult,
                result_container="object",
            )


class TestValidateParameterCount:
    def test_valid_parameter_count_passes(self) -> None:
        def handler(
            params: SampleParams, *, context: ToolContext
        ) -> ToolResult[SampleResult]:
            return ToolResult.ok(SampleResult(value="ok"), message="ok")

        validator = ToolValidator()
        sig = inspect.signature(handler)
        params = list(sig.parameters.values())
        param, context_param = validator.validate_parameter_count(
            params, params_type=SampleParams
        )
        assert param.name == "params"
        assert context_param.name == "context"

    def test_wrong_parameter_count_fails(self) -> None:
        def handler(params: SampleParams) -> ToolResult[SampleResult]:
            return ToolResult.ok(SampleResult(value="ok"), message="ok")

        validator = ToolValidator()
        sig = inspect.signature(handler)
        params = list(sig.parameters.values())
        with pytest.raises(PromptValidationError, match="one positional argument"):
            validator.validate_parameter_count(params, params_type=SampleParams)


# =============================================================================
# TypeResolver tests
# =============================================================================


class TestNormalizeResultAnnotation:
    def test_none_annotation_returns_none_type(self) -> None:
        resolver = TypeResolver()
        result_type, container = resolver.normalize_result_annotation(
            None, params_type=SampleParams
        )
        assert result_type is type(None)
        assert container == "object"

    def test_type_annotation_returns_object(self) -> None:
        resolver = TypeResolver()
        result_type, container = resolver.normalize_result_annotation(
            SampleResult, params_type=SampleParams
        )
        assert result_type is SampleResult
        assert container == "object"

    def test_list_annotation_returns_array(self) -> None:
        resolver = TypeResolver()
        result_type, container = resolver.normalize_result_annotation(
            list[SampleResult], params_type=SampleParams
        )
        assert result_type is SampleResult
        assert container == "array"

    def test_sequence_annotation_returns_array(self) -> None:
        resolver = TypeResolver()
        result_type, container = resolver.normalize_result_annotation(
            Sequence[SampleResult], params_type=SampleParams
        )
        assert result_type is SampleResult
        assert container == "array"

    def test_variadic_tuple_annotation_returns_array(self) -> None:
        resolver = TypeResolver()
        result_type, container = resolver.normalize_result_annotation(
            tuple[SampleResult, ...], params_type=SampleParams
        )
        assert result_type is SampleResult
        assert container == "array"

    def test_fixed_tuple_annotation_fails(self) -> None:
        resolver = TypeResolver()
        with pytest.raises(PromptValidationError, match="Variadic Tuple"):
            resolver.normalize_result_annotation(
                tuple[SampleResult, SampleResult], params_type=SampleParams
            )

    def test_invalid_origin_fails(self) -> None:
        resolver = TypeResolver()
        with pytest.raises(PromptValidationError, match="dataclass type"):
            resolver.normalize_result_annotation(
                dict[str, SampleResult], params_type=SampleParams
            )

    def test_non_type_element_fails(self) -> None:
        resolver = TypeResolver()
        with pytest.raises(PromptValidationError, match="dataclass type"):
            resolver.normalize_result_annotation(
                list["SampleResult"],
                params_type=SampleParams,  # type: ignore[arg-type]
            )


class TestResolveWrappedDescription:
    def test_extracts_docstring(self) -> None:
        def handler(
            params: SampleParams, *, context: ToolContext
        ) -> ToolResult[SampleResult]:
            """Does something useful."""
            return ToolResult.ok(SampleResult(value="ok"), message="ok")

        resolver = TypeResolver()
        description = resolver.resolve_wrapped_description(handler)
        assert description == "Does something useful."

    def test_missing_docstring_fails(self) -> None:
        def handler(
            params: SampleParams, *, context: ToolContext
        ) -> ToolResult[SampleResult]:
            return ToolResult.ok(SampleResult(value="ok"), message="ok")

        resolver = TypeResolver()
        with pytest.raises(PromptValidationError, match="docstring"):
            resolver.resolve_wrapped_description(handler)


class TestResolveWrappedParamsType:
    def test_extracts_params_type(self) -> None:
        def handler(
            params: SampleParams, *, context: ToolContext
        ) -> ToolResult[SampleResult]:
            """Handler."""
            return ToolResult.ok(SampleResult(value="ok"), message="ok")

        resolver = TypeResolver()
        sig = inspect.signature(handler)
        param = next(iter(sig.parameters.values()))
        hints = resolver.resolve_annotations(handler)
        params_type = resolver.resolve_wrapped_params_type(param, hints)
        assert params_type is SampleParams

    def test_unwraps_annotated(self) -> None:
        def handler(
            params: Annotated[SampleParams, "meta"], *, context: ToolContext
        ) -> ToolResult[SampleResult]:
            """Handler."""
            return ToolResult.ok(SampleResult(value="ok"), message="ok")

        resolver = TypeResolver()
        sig = inspect.signature(handler)
        param = next(iter(sig.parameters.values()))
        hints = resolver.resolve_annotations(handler)
        params_type = resolver.resolve_wrapped_params_type(param, hints)
        assert params_type is SampleParams

    def test_missing_annotation_fails(self) -> None:
        def handler(
            params,  # noqa: ANN001
            *,
            context: ToolContext,
        ) -> ToolResult[SampleResult]:  # type: ignore[no-untyped-def]
            """Handler."""
            return ToolResult.ok(SampleResult(value="ok"), message="ok")

        resolver = TypeResolver()
        sig = inspect.signature(handler)
        param = next(iter(sig.parameters.values()))
        hints = resolver.resolve_annotations(handler)
        with pytest.raises(PromptValidationError, match="annotated with ParamsT"):
            resolver.resolve_wrapped_params_type(param, hints)


class TestResolveWrappedResultAnnotation:
    def test_extracts_result_type(self) -> None:
        def handler(
            params: SampleParams, *, context: ToolContext
        ) -> ToolResult[SampleResult]:
            """Handler."""
            return ToolResult.ok(SampleResult(value="ok"), message="ok")

        resolver = TypeResolver()
        sig = inspect.signature(handler)
        hints = resolver.resolve_annotations(handler)
        result = resolver.resolve_wrapped_result_annotation(
            sig, hints, params_type=SampleParams
        )
        assert result is SampleResult

    def test_unwraps_annotated_return(self) -> None:
        def handler(
            params: SampleParams, *, context: ToolContext
        ) -> Annotated[ToolResult[SampleResult], "meta"]:
            """Handler."""
            return ToolResult.ok(SampleResult(value="ok"), message="ok")

        resolver = TypeResolver()
        sig = inspect.signature(handler)
        hints = resolver.resolve_annotations(handler)
        result = resolver.resolve_wrapped_result_annotation(
            sig, hints, params_type=SampleParams
        )
        assert result is SampleResult

    def test_missing_return_annotation_fails(self) -> None:
        def handler(  # noqa: ANN202
            params: SampleParams, *, context: ToolContext
        ):  # type: ignore[no-untyped-def]
            """Handler."""
            return ToolResult.ok(SampleResult(value="ok"), message="ok")

        resolver = TypeResolver()
        sig = inspect.signature(handler)
        hints = resolver.resolve_annotations(handler)
        with pytest.raises(PromptValidationError, match="ToolResult"):
            resolver.resolve_wrapped_result_annotation(
                sig, hints, params_type=SampleParams
            )

    def test_non_toolresult_return_fails(self) -> None:
        def handler(params: SampleParams, *, context: ToolContext) -> SampleResult:  # type: ignore[type-arg]
            """Handler."""
            return SampleResult(value="ok")

        resolver = TypeResolver()
        sig = inspect.signature(handler)
        hints = resolver.resolve_annotations(handler)
        with pytest.raises(PromptValidationError, match="ToolResult"):
            resolver.resolve_wrapped_result_annotation(
                sig, hints, params_type=SampleParams
            )


# =============================================================================
# Helper function tests
# =============================================================================


class TestCoerceNoneType:
    def test_none_becomes_none_type(self) -> None:
        assert coerce_none_type(None) is type(None)

    def test_type_unchanged(self) -> None:
        assert coerce_none_type(SampleParams) is SampleParams

    def test_union_with_none_extracts_non_none(self) -> None:
        result = coerce_none_type(SampleParams | None)
        assert result is SampleParams


class TestNormalizeSpecialization:
    def test_valid_tuple_passes(self) -> None:
        params, result = normalize_specialization((SampleParams, SampleResult))
        assert params is SampleParams
        assert result is SampleResult

    def test_none_values_coerced(self) -> None:
        params, result = normalize_specialization((None, None))
        assert params is type(None)
        assert result is type(None)

    def test_non_tuple_fails(self) -> None:
        with pytest.raises(TypeError, match="two type arguments"):
            normalize_specialization(SampleParams)

    def test_wrong_length_fails(self) -> None:
        with pytest.raises(TypeError, match="two type arguments"):
            normalize_specialization((SampleParams,))


# =============================================================================
# create_tool factory function tests
# =============================================================================


class TestCreateTool:
    def test_creates_tool_from_handler(self) -> None:
        def search(
            params: SampleParams, *, context: ToolContext
        ) -> ToolResult[SampleResult]:
            """Search for items."""
            return ToolResult.ok(SampleResult(value=params.query), message="ok")

        tool = create_tool("search", search)

        assert tool.name == "search"
        assert tool.description == "Search for items."
        assert tool.handler is search
        assert tool.params_type is SampleParams
        assert tool.result_type is SampleResult

    def test_custom_description_overrides_docstring(self) -> None:
        def search(
            params: SampleParams, *, context: ToolContext
        ) -> ToolResult[SampleResult]:
            """Original description."""
            return ToolResult.ok(SampleResult(value=params.query), message="ok")

        tool = create_tool("search", search, description="Custom description.")

        assert tool.description == "Custom description."

    def test_accepts_examples(self) -> None:
        def search(
            params: SampleParams, *, context: ToolContext
        ) -> ToolResult[SampleResult]:
            """Search for items."""
            return ToolResult.ok(SampleResult(value=params.query), message="ok")

        example = ToolExample[SampleParams, SampleResult](
            description="simple search",
            input=SampleParams(query="test"),
            output=SampleResult(value="result"),
        )
        tool = create_tool("search", search, examples=(example,))

        assert tool.examples == (example,)

    def test_accepts_overrides_flag(self) -> None:
        def search(
            params: SampleParams, *, context: ToolContext
        ) -> ToolResult[SampleResult]:
            """Search for items."""
            return ToolResult.ok(SampleResult(value=params.query), message="ok")

        tool = create_tool("search", search, accepts_overrides=False)

        assert tool.accepts_overrides is False
