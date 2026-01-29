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

"""Tests for scoped field visibility (HiddenInStructuredOutput)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, cast

import pytest

from weakincentives.serde import (
    HiddenInStructuredOutput,
    SerdeScope,
    dump,
    parse,
    schema,
)
from weakincentives.serde._scope import is_hidden_in_scope
from weakincentives.types import JSONValue

pytestmark = pytest.mark.core


# =============================================================================
# Test Fixtures
# =============================================================================


@dataclass
class AnalysisResult:
    """Example dataclass with hidden fields for testing."""

    summary: str
    confidence: float

    # Hidden from LLM schema - populated in finalize()
    processing_time_ms: Annotated[int, HiddenInStructuredOutput()] = 0
    model_version: Annotated[str, HiddenInStructuredOutput()] = ""


@dataclass
class NestedHidden:
    """Dataclass with nested hidden field."""

    @dataclass
    class Metadata:
        timestamp: str
        source: str

    content: str
    meta: Annotated[Metadata, HiddenInStructuredOutput()] = field(
        default_factory=lambda: NestedHidden.Metadata("", "system")
    )


@dataclass
class MixedConstraints:
    """Hidden field with additional constraints."""

    value: int
    score: Annotated[int, HiddenInStructuredOutput(), {"ge": 0, "le": 100}] = 0
    label: Annotated[str, HiddenInStructuredOutput(), {"alias": "tag"}] = ""


@dataclass
class AllHidden:
    """Dataclass with all fields hidden (edge case)."""

    hidden_a: Annotated[int, HiddenInStructuredOutput()] = 0
    hidden_b: Annotated[str, HiddenInStructuredOutput()] = ""


@dataclass
class NoHidden:
    """Dataclass with no hidden fields."""

    visible_a: int
    visible_b: str


# =============================================================================
# is_hidden_in_scope Tests
# =============================================================================


def test_is_hidden_in_scope_default_scope_never_hides() -> None:
    """DEFAULT scope never hides fields."""
    hidden_type = Annotated[int, HiddenInStructuredOutput()]
    assert is_hidden_in_scope(hidden_type, SerdeScope.DEFAULT) is False


def test_is_hidden_in_scope_structured_output_hides_marked() -> None:
    """STRUCTURED_OUTPUT scope hides fields with HiddenInStructuredOutput marker."""
    hidden_type = Annotated[int, HiddenInStructuredOutput()]
    assert is_hidden_in_scope(hidden_type, SerdeScope.STRUCTURED_OUTPUT) is True


def test_is_hidden_in_scope_plain_type_not_hidden() -> None:
    """Plain types without Annotated are not hidden."""
    assert is_hidden_in_scope(int, SerdeScope.STRUCTURED_OUTPUT) is False
    assert is_hidden_in_scope(str, SerdeScope.STRUCTURED_OUTPUT) is False


def test_is_hidden_in_scope_annotated_without_marker() -> None:
    """Annotated types without HiddenInStructuredOutput are not hidden."""
    constraint_type = Annotated[int, {"ge": 0}]
    assert is_hidden_in_scope(constraint_type, SerdeScope.STRUCTURED_OUTPUT) is False


def test_is_hidden_in_scope_with_multiple_annotations() -> None:
    """HiddenInStructuredOutput works with other annotations."""
    mixed_type = Annotated[int, {"ge": 0}, HiddenInStructuredOutput(), {"le": 100}]
    assert is_hidden_in_scope(mixed_type, SerdeScope.STRUCTURED_OUTPUT) is True


# =============================================================================
# schema() with scope Tests
# =============================================================================


def test_schema_default_scope_includes_all_fields() -> None:
    """schema() with DEFAULT scope includes all fields."""
    schema_dict = schema(AnalysisResult, scope=SerdeScope.DEFAULT)
    properties = cast(dict[str, JSONValue], schema_dict["properties"])

    assert "summary" in properties
    assert "confidence" in properties
    assert "processing_time_ms" in properties
    assert "model_version" in properties


def test_schema_structured_output_scope_excludes_hidden() -> None:
    """schema() with STRUCTURED_OUTPUT scope excludes hidden fields."""
    schema_dict = schema(AnalysisResult, scope=SerdeScope.STRUCTURED_OUTPUT)
    properties = cast(dict[str, JSONValue], schema_dict["properties"])

    assert "summary" in properties
    assert "confidence" in properties
    assert "processing_time_ms" not in properties
    assert "model_version" not in properties


def test_schema_default_scope_is_default_behavior() -> None:
    """schema() defaults to DEFAULT scope."""
    schema_with_default = schema(AnalysisResult)
    schema_explicit_default = schema(AnalysisResult, scope=SerdeScope.DEFAULT)

    assert schema_with_default == schema_explicit_default


def test_schema_required_excludes_hidden_in_structured_output() -> None:
    """Required list excludes hidden fields in STRUCTURED_OUTPUT scope."""

    @dataclass
    class RequiredHidden:
        visible: str
        hidden: Annotated[int, HiddenInStructuredOutput()] = 0

    schema_dict = schema(RequiredHidden, scope=SerdeScope.STRUCTURED_OUTPUT)
    required = cast(list[str], schema_dict.get("required", []))

    assert "visible" in required
    assert "hidden" not in required


def test_schema_all_hidden_produces_empty_properties() -> None:
    """Dataclass with all hidden fields has empty properties in STRUCTURED_OUTPUT."""
    schema_dict = schema(AllHidden, scope=SerdeScope.STRUCTURED_OUTPUT)
    properties = cast(dict[str, JSONValue], schema_dict["properties"])

    assert properties == {}
    assert "required" not in schema_dict


def test_schema_nested_hidden_excludes_entire_subtree() -> None:
    """Hidden field with nested dataclass excludes entire schema subtree."""
    schema_dict = schema(NestedHidden, scope=SerdeScope.STRUCTURED_OUTPUT)
    properties = cast(dict[str, JSONValue], schema_dict["properties"])

    assert "content" in properties
    assert "meta" not in properties


def test_schema_no_hidden_same_in_both_scopes() -> None:
    """Dataclass with no hidden fields produces same schema in both scopes."""
    schema_default = schema(NoHidden, scope=SerdeScope.DEFAULT)
    schema_structured = schema(NoHidden, scope=SerdeScope.STRUCTURED_OUTPUT)

    assert schema_default == schema_structured


def test_schema_hidden_field_without_default_behavior() -> None:
    """Hidden field without default: schema succeeds, parse fails naturally."""

    @dataclass
    class InvalidHidden:
        visible: str
        hidden_no_default: Annotated[int, HiddenInStructuredOutput()]  # No default!

    # DEFAULT scope - field is visible and required
    schema_default = schema(InvalidHidden, scope=SerdeScope.DEFAULT)
    properties = cast(dict[str, object], schema_default["properties"])
    assert "hidden_no_default" in properties
    required = cast(list[str], schema_default.get("required", []))
    assert "hidden_no_default" in required

    # STRUCTURED_OUTPUT scope - field is excluded from schema
    schema_structured = schema(InvalidHidden, scope=SerdeScope.STRUCTURED_OUTPUT)
    properties_structured = cast(dict[str, object], schema_structured["properties"])
    assert "hidden_no_default" not in properties_structured

    # Parse fails at instantiation time with natural dataclass error
    with pytest.raises(TypeError) as exc:
        parse(InvalidHidden, {"visible": "test"}, scope=SerdeScope.STRUCTURED_OUTPUT)
    # Python's dataclass __init__ raises: "missing 1 required positional argument"
    assert "hidden_no_default" in str(exc.value)


# =============================================================================
# parse() with scope Tests
# =============================================================================


def test_parse_default_scope_includes_all_fields() -> None:
    """parse() with DEFAULT scope parses all fields."""
    data = {
        "summary": "test",
        "confidence": 0.95,
        "processing_time_ms": 100,
        "model_version": "v1.0",
    }
    result = parse(AnalysisResult, data, scope=SerdeScope.DEFAULT)

    assert result.summary == "test"
    assert result.confidence == 0.95
    assert result.processing_time_ms == 100
    assert result.model_version == "v1.0"


def test_parse_structured_output_scope_skips_hidden() -> None:
    """parse() with STRUCTURED_OUTPUT scope skips hidden fields, uses defaults."""
    data = {
        "summary": "test",
        "confidence": 0.95,
        "processing_time_ms": 100,  # Present but should be ignored
        "model_version": "v1.0",  # Present but should be ignored
    }
    result = parse(AnalysisResult, data, scope=SerdeScope.STRUCTURED_OUTPUT)

    assert result.summary == "test"
    assert result.confidence == 0.95
    # Hidden fields use defaults, not the provided values
    assert result.processing_time_ms == 0
    assert result.model_version == ""


def test_parse_default_scope_is_default_behavior() -> None:
    """parse() defaults to DEFAULT scope."""
    data = {
        "summary": "test",
        "confidence": 0.95,
        "processing_time_ms": 100,
        "model_version": "v1.0",
    }
    result_default = parse(AnalysisResult, data)
    result_explicit = parse(AnalysisResult, data, scope=SerdeScope.DEFAULT)

    assert result_default == result_explicit


def test_parse_structured_output_hidden_fields_use_defaults() -> None:
    """Hidden fields use their default values in STRUCTURED_OUTPUT scope."""

    @dataclass
    class WithDefaults:
        visible: str
        hidden_int: Annotated[int, HiddenInStructuredOutput()] = 42
        hidden_str: Annotated[str, HiddenInStructuredOutput()] = "default"

    data = {"visible": "test", "hidden_int": 999, "hidden_str": "overridden"}
    result = parse(WithDefaults, data, scope=SerdeScope.STRUCTURED_OUTPUT)

    assert result.visible == "test"
    assert result.hidden_int == 42  # Default, not 999
    assert result.hidden_str == "default"  # Default, not "overridden"


def test_parse_structured_output_hidden_field_with_factory() -> None:
    """Hidden field with default_factory uses factory in STRUCTURED_OUTPUT scope."""

    @dataclass
    class WithFactory:
        visible: str
        hidden: Annotated[list[int], HiddenInStructuredOutput()] = field(
            default_factory=lambda: [1, 2, 3]
        )

    data = {"visible": "test", "hidden": [4, 5, 6]}
    result = parse(WithFactory, data, scope=SerdeScope.STRUCTURED_OUTPUT)

    assert result.visible == "test"
    assert result.hidden == [1, 2, 3]  # Factory default, not [4, 5, 6]


def test_parse_all_hidden_requires_defaults() -> None:
    """All-hidden dataclass must have defaults for STRUCTURED_OUTPUT parsing."""
    data: dict[str, object] = {}  # No data provided
    result = parse(AllHidden, data, scope=SerdeScope.STRUCTURED_OUTPUT)

    assert result.hidden_a == 0
    assert result.hidden_b == ""


# =============================================================================
# dump() Tests (Always Includes Hidden)
# =============================================================================


def test_dump_always_includes_hidden_fields() -> None:
    """dump() always includes hidden fields regardless of any scope."""
    result = AnalysisResult(
        summary="test",
        confidence=0.95,
        processing_time_ms=100,
        model_version="v1.0",
    )
    data = dump(result)

    assert data["summary"] == "test"
    assert data["confidence"] == 0.95
    assert data["processing_time_ms"] == 100
    assert data["model_version"] == "v1.0"


def test_dump_hidden_fields_with_defaults() -> None:
    """dump() includes hidden fields even when they have default values."""
    result = AnalysisResult(summary="test", confidence=0.95)
    data = dump(result)

    assert data["processing_time_ms"] == 0
    assert data["model_version"] == ""


# =============================================================================
# Integration Tests
# =============================================================================


def test_finalize_pattern_round_trip() -> None:
    """Simulate the finalize() pattern: parse with STRUCTURED_OUTPUT, then replace."""
    from dataclasses import replace

    # LLM response only contains visible fields
    llm_response = {"summary": "Analysis complete", "confidence": 0.87}

    # Parse with STRUCTURED_OUTPUT scope (as done in response parser)
    result = parse(AnalysisResult, llm_response, scope=SerdeScope.STRUCTURED_OUTPUT)

    assert result.summary == "Analysis complete"
    assert result.confidence == 0.87
    assert result.processing_time_ms == 0  # Default
    assert result.model_version == ""  # Default

    # Simulate finalize() populating hidden fields
    finalized = replace(result, processing_time_ms=150, model_version="gpt-4")

    assert finalized.processing_time_ms == 150
    assert finalized.model_version == "gpt-4"

    # dump() captures everything
    data = dump(finalized)
    assert data["processing_time_ms"] == 150
    assert data["model_version"] == "gpt-4"


def test_constraints_still_apply_to_hidden_fields_in_default_scope() -> None:
    """Constraints on hidden fields are enforced in DEFAULT scope."""
    # Valid in DEFAULT scope
    valid_data = {"value": 1, "score": 50, "label": "test"}
    result = parse(MixedConstraints, valid_data, scope=SerdeScope.DEFAULT)
    assert result.score == 50

    # Invalid score (violates ge=0, le=100) in DEFAULT scope
    invalid_data = {"value": 1, "score": 150, "label": "test"}
    with pytest.raises(ValueError) as exc:
        parse(MixedConstraints, invalid_data, scope=SerdeScope.DEFAULT)
    assert "score: must be <= 100" in str(exc.value)


def test_hidden_fields_skipped_in_structured_output_regardless_of_constraints() -> None:
    """Hidden fields are skipped entirely in STRUCTURED_OUTPUT scope."""
    # Even with invalid data for hidden fields, they're skipped
    data = {"value": 1, "score": 999, "label": "invalid"}  # Would violate constraints
    result = parse(MixedConstraints, data, scope=SerdeScope.STRUCTURED_OUTPUT)

    assert result.value == 1
    assert result.score == 0  # Default, constraints not checked
    assert result.label == ""  # Default


def test_hidden_field_keys_in_data_are_extra_in_structured_output() -> None:
    """Hidden field keys in input data are treated as extra keys in STRUCTURED_OUTPUT."""
    data = {
        "summary": "test",
        "confidence": 0.9,
        "processing_time_ms": 100,  # Hidden field key
    }

    # With extra='forbid', hidden field key in data raises error
    with pytest.raises(ValueError) as exc:
        parse(AnalysisResult, data, scope=SerdeScope.STRUCTURED_OUTPUT, extra="forbid")
    assert "processing_time_ms" in str(exc.value)

    # With extra='ignore' (default), hidden field key in data is silently ignored
    result = parse(AnalysisResult, data, scope=SerdeScope.STRUCTURED_OUTPUT)
    assert result.processing_time_ms == 0  # Default, not 100
