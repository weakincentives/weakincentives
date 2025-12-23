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

"""Property-based tests for dataclass serde."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Annotated, Literal
from uuid import UUID

import pytest
from hypothesis import given, settings, strategies as st

from weakincentives.serde import clone, dump, parse, schema


# ============================================================================
# Test Fixtures: Dataclass Definitions
# ============================================================================


class Priority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True, slots=True)
class SimpleRecord:
    """Simple dataclass with primitives."""

    name: str
    count: int
    active: bool
    score: float = 0.0


@dataclass(frozen=True, slots=True)
class ConstrainedRecord:
    """Dataclass with field constraints."""

    text: Annotated[str, {"min_length": 1, "strip": True}]
    value: Annotated[int, {"ge": 0, "le": 100}]
    tags: list[Annotated[str, {"regex": r"^tag-[a-z]+$"}]]


@dataclass(frozen=True, slots=True)
class NestedNote:
    """Nested dataclass with constraints."""

    text: Annotated[str, {"min_length": 1, "strip": True}]
    tags: list[Annotated[str, {"regex": r"^tag-[a-z]+$"}]]


@dataclass(frozen=True, slots=True)
class NestedRecord:
    """Dataclass with nested structures."""

    code: Annotated[str, {"regex": r"^REC-\d{3}$"}]
    note: NestedNote
    metrics: dict[str, Annotated[int | None, {"ge": 0, "le": 5}]]
    optional_notes: list[NestedNote | None]
    rating: Annotated[int | None, {"ge": 1, "le": 10}] = None


@dataclass(frozen=True, slots=True)
class TemporalRecord:
    """Dataclass with temporal types."""

    created_at: datetime
    due_date: date
    start_time: time


@dataclass(frozen=True, slots=True)
class SpecialTypesRecord:
    """Dataclass with special types: UUID, Decimal, Path."""

    id: UUID
    amount: Decimal
    filepath: Path


@dataclass(frozen=True, slots=True)
class EnumRecord:
    """Dataclass with enum fields."""

    priority: Priority
    optional_priority: Priority | None = None


@dataclass(frozen=True, slots=True)
class LiteralRecord:
    """Dataclass with literal types."""

    status: Literal["pending", "active", "completed"]
    level: Literal[1, 2, 3]


@dataclass(frozen=True, slots=True)
class CollectionRecord:
    """Dataclass with various collection types."""

    items: list[str]
    unique_items: set[int]
    mapping: dict[str, int]
    fixed_tuple: tuple[str, int, bool]


@dataclass(frozen=True, slots=True)
class OptionalFieldsRecord:
    """Dataclass with optional fields."""

    required: str
    optional_str: str | None = None
    optional_int: int | None = None
    optional_list: list[str] | None = None


@dataclass(frozen=True, slots=True)
class AliasedRecord:
    """Dataclass with aliased fields."""

    user_name: str = field(metadata={"alias": "userName"})
    email_address: str = field(metadata={"alias": "emailAddress"})


@dataclass(frozen=True, slots=True)
class ValidatedRecord:
    """Dataclass with validation hooks."""

    value: int

    def __validate__(self) -> None:
        if self.value < 0:
            msg = "value must be non-negative"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ComputedRecord:
    """Dataclass with computed fields."""

    first_name: str
    last_name: str

    __computed__ = ("full_name",)

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"


@dataclass(slots=True)
class MutableRecord:
    """Mutable dataclass for clone testing."""

    name: str
    count: int = 0


# ============================================================================
# Hypothesis Strategies
# ============================================================================

_tag_texts = st.text(
    alphabet=st.sampled_from(tuple("abcdefghijklmnopqrstuvwxyz")),
    min_size=1,
    max_size=8,
)

_safe_text = st.text(
    alphabet=st.characters(
        min_codepoint=32,
        max_codepoint=126,
        categories=["L", "N", "P", "Zs"],
    ),
    min_size=0,
    max_size=50,
)

_nonempty_text = st.text(
    alphabet=st.characters(min_codepoint=32, max_codepoint=126),
    min_size=1,
    max_size=20,
)


def simple_record_strategy() -> st.SearchStrategy[SimpleRecord]:
    return st.builds(
        SimpleRecord,
        name=_safe_text,
        count=st.integers(min_value=-1000, max_value=1000),
        active=st.booleans(),
        score=st.floats(
            min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
        ),
    )


def constrained_record_strategy() -> st.SearchStrategy[ConstrainedRecord]:
    return st.builds(
        ConstrainedRecord,
        text=_nonempty_text,
        value=st.integers(min_value=0, max_value=100),
        tags=st.lists(
            st.from_regex(r"^tag-[a-z]+$", fullmatch=True),
            min_size=0,
            max_size=3,
        ),
    )


def note_strategy() -> st.SearchStrategy[NestedNote]:
    tag_strategy = st.from_regex(r"^tag-[a-z]+$", fullmatch=True)
    return st.builds(
        NestedNote,
        text=_tag_texts,
        tags=st.lists(tag_strategy, min_size=1, max_size=3),
    )


def record_strategy() -> st.SearchStrategy[NestedRecord]:
    return st.builds(
        NestedRecord,
        code=st.from_regex(r"^REC-\d{3}$", fullmatch=True),
        note=note_strategy(),
        metrics=st.dictionaries(
            st.sampled_from(["alpha", "beta", "gamma"]),
            st.one_of(st.integers(min_value=0, max_value=5), st.none()),
            min_size=1,
            max_size=3,
        ),
        optional_notes=st.lists(st.one_of(note_strategy(), st.none()), max_size=2),
        rating=st.one_of(st.integers(min_value=1, max_value=10), st.none()),
    )


def temporal_record_strategy() -> st.SearchStrategy[TemporalRecord]:
    return st.builds(
        TemporalRecord,
        created_at=st.datetimes(
            min_value=datetime(2000, 1, 1),
            max_value=datetime(2100, 1, 1),
            timezones=st.just(timezone.utc),
        ),
        due_date=st.dates(min_value=date(2000, 1, 1), max_value=date(2100, 1, 1)),
        start_time=st.times(),
    )


def special_types_strategy() -> st.SearchStrategy[SpecialTypesRecord]:
    return st.builds(
        SpecialTypesRecord,
        id=st.uuids(),
        amount=st.decimals(
            min_value=Decimal("-1e6"),
            max_value=Decimal("1e6"),
            allow_nan=False,
            allow_infinity=False,
            places=2,
        ),
        filepath=st.builds(
            Path,
            st.text(
                alphabet=st.characters(
                    min_codepoint=97, max_codepoint=122, categories=["Ll"]
                ),
                min_size=1,
                max_size=20,
            ),
        ),
    )


def enum_record_strategy() -> st.SearchStrategy[EnumRecord]:
    return st.builds(
        EnumRecord,
        priority=st.sampled_from(list(Priority)),
        optional_priority=st.one_of(st.sampled_from(list(Priority)), st.none()),
    )


def literal_record_strategy() -> st.SearchStrategy[LiteralRecord]:
    return st.builds(
        LiteralRecord,
        status=st.sampled_from(["pending", "active", "completed"]),
        level=st.sampled_from([1, 2, 3]),
    )


def collection_record_strategy() -> st.SearchStrategy[CollectionRecord]:
    return st.builds(
        CollectionRecord,
        items=st.lists(_nonempty_text, max_size=5),
        unique_items=st.frozensets(st.integers(min_value=0, max_value=100), max_size=5),
        mapping=st.dictionaries(_nonempty_text, st.integers(), max_size=3),
        fixed_tuple=st.tuples(_nonempty_text, st.integers(), st.booleans()),
    )


def optional_fields_strategy() -> st.SearchStrategy[OptionalFieldsRecord]:
    return st.builds(
        OptionalFieldsRecord,
        required=_nonempty_text,
        optional_str=st.one_of(_nonempty_text, st.none()),
        optional_int=st.one_of(st.integers(), st.none()),
        optional_list=st.one_of(st.lists(_nonempty_text, max_size=3), st.none()),
    )


def aliased_record_strategy() -> st.SearchStrategy[AliasedRecord]:
    return st.builds(
        AliasedRecord,
        user_name=_nonempty_text,
        email_address=_nonempty_text,
    )


def validated_record_strategy() -> st.SearchStrategy[ValidatedRecord]:
    return st.builds(
        ValidatedRecord,
        value=st.integers(min_value=0, max_value=1000),
    )


def computed_record_strategy() -> st.SearchStrategy[ComputedRecord]:
    return st.builds(
        ComputedRecord,
        first_name=_nonempty_text,
        last_name=_nonempty_text,
    )


def mutable_record_strategy() -> st.SearchStrategy[MutableRecord]:
    return st.builds(
        MutableRecord,
        name=_nonempty_text,
        count=st.integers(min_value=0, max_value=100),
    )


# ============================================================================
# Property Tests: dump/parse Roundtrip
# ============================================================================


@given(simple_record_strategy())
@settings(max_examples=100)
def test_roundtrip_simple_record(record: SimpleRecord) -> None:
    """Simple dataclass survives dump→parse cycle."""
    payload = dump(record)
    restored = parse(SimpleRecord, payload)
    assert restored == record


@given(constrained_record_strategy())
@settings(max_examples=100)
def test_roundtrip_constrained_record(record: ConstrainedRecord) -> None:
    """Constrained dataclass survives dump→parse cycle."""
    payload = dump(record)
    restored = parse(ConstrainedRecord, payload)
    assert restored == record


@given(record_strategy())
@settings(max_examples=100)
def test_roundtrip_with_nested_collections(record: NestedRecord) -> None:
    """Nested dataclass survives dump→parse cycle."""
    payload = dump(record)
    restored = parse(NestedRecord, payload)
    assert restored == record


@given(temporal_record_strategy())
@settings(max_examples=100)
def test_roundtrip_temporal_types(record: TemporalRecord) -> None:
    """Temporal types (datetime, date, time) survive dump→parse cycle."""
    payload = dump(record)
    restored = parse(TemporalRecord, payload)
    assert restored == record


@given(special_types_strategy())
@settings(max_examples=100)
def test_roundtrip_special_types(record: SpecialTypesRecord) -> None:
    """Special types (UUID, Decimal, Path) survive dump→parse cycle."""
    payload = dump(record)
    restored = parse(SpecialTypesRecord, payload)
    assert restored == record


@given(enum_record_strategy())
@settings(max_examples=100)
def test_roundtrip_enum_types(record: EnumRecord) -> None:
    """Enum types survive dump→parse cycle."""
    payload = dump(record)
    restored = parse(EnumRecord, payload)
    assert restored == record


@given(literal_record_strategy())
@settings(max_examples=100)
def test_roundtrip_literal_types(record: LiteralRecord) -> None:
    """Literal types survive dump→parse cycle."""
    payload = dump(record)
    restored = parse(LiteralRecord, payload)
    assert restored == record


@given(collection_record_strategy())
@settings(max_examples=100)
def test_roundtrip_collection_types(record: CollectionRecord) -> None:
    """Collection types (list, set, dict, tuple) survive dump→parse cycle."""
    payload = dump(record)
    restored = parse(CollectionRecord, payload)
    # Sets are serialized as sorted lists, so compare elements
    assert restored.items == record.items
    assert restored.unique_items == record.unique_items
    assert restored.mapping == record.mapping
    assert restored.fixed_tuple == record.fixed_tuple


@given(optional_fields_strategy())
@settings(max_examples=100)
def test_roundtrip_optional_fields(record: OptionalFieldsRecord) -> None:
    """Optional fields survive dump→parse cycle."""
    payload = dump(record)
    restored = parse(OptionalFieldsRecord, payload)
    assert restored == record


@given(aliased_record_strategy())
@settings(max_examples=100)
def test_roundtrip_aliased_fields(record: AliasedRecord) -> None:
    """Aliased fields are serialized with aliases and parsed back correctly."""
    payload = dump(record, by_alias=True)
    assert "userName" in payload
    assert "emailAddress" in payload
    restored = parse(AliasedRecord, payload)
    assert restored == record


@given(validated_record_strategy())
@settings(max_examples=100)
def test_roundtrip_validated_record(record: ValidatedRecord) -> None:
    """Validated dataclass survives dump→parse cycle."""
    payload = dump(record)
    restored = parse(ValidatedRecord, payload)
    assert restored == record


@given(computed_record_strategy())
@settings(max_examples=100)
def test_roundtrip_computed_record(record: ComputedRecord) -> None:
    """Computed fields are included when computed=True."""
    payload = dump(record, computed=True)
    assert "full_name" in payload
    assert payload["full_name"] == record.full_name
    # Parse back (without computed, since it's derived)
    restored = parse(ComputedRecord, {"first_name": record.first_name, "last_name": record.last_name})
    assert restored == record


# ============================================================================
# Property Tests: clone
# ============================================================================


@given(simple_record_strategy())
@settings(max_examples=100)
def test_clone_preserves_simple_record(record: SimpleRecord) -> None:
    """Clone without updates preserves equality."""
    cloned = clone(record)
    assert cloned == record
    assert cloned is not record


@given(constrained_record_strategy())
@settings(max_examples=100)
def test_clone_preserves_constrained_record(record: ConstrainedRecord) -> None:
    """Clone preserves constrained record."""
    cloned = clone(record)
    assert cloned == record


@given(simple_record_strategy(), _nonempty_text)
@settings(max_examples=100)
def test_clone_with_updates(record: SimpleRecord, new_name: str) -> None:
    """Clone with updates produces correct result."""
    cloned = clone(record, name=new_name)
    assert cloned.name == new_name
    assert cloned.count == record.count
    assert cloned.active == record.active
    assert cloned.score == record.score


@given(validated_record_strategy())
@settings(max_examples=100)
def test_clone_reruns_validation(record: ValidatedRecord) -> None:
    """Clone re-runs __validate__ hook."""
    # This should succeed since strategy generates valid values
    cloned = clone(record)
    assert cloned == record


@given(mutable_record_strategy())
@settings(max_examples=100)
def test_clone_mutable_record(record: MutableRecord) -> None:
    """Clone works with mutable dataclasses."""
    cloned = clone(record)
    assert cloned.name == record.name
    assert cloned.count == record.count


# ============================================================================
# Property Tests: schema
# ============================================================================


@given(st.sampled_from([
    SimpleRecord,
    ConstrainedRecord,
    NestedNote,
    NestedRecord,
    TemporalRecord,
    SpecialTypesRecord,
    EnumRecord,
    LiteralRecord,
    CollectionRecord,
    OptionalFieldsRecord,
    AliasedRecord,
    ValidatedRecord,
    ComputedRecord,
]))
@settings(max_examples=50)
def test_schema_is_valid_json_schema(cls: type[object]) -> None:
    """Generated schema is a valid JSON Schema structure."""
    schema_dict = schema(cls)
    assert isinstance(schema_dict, dict)
    assert "type" in schema_dict
    assert schema_dict["type"] == "object"
    assert "properties" in schema_dict
    assert isinstance(schema_dict["properties"], dict)
    assert "title" in schema_dict
    assert schema_dict["title"] == cls.__name__


@given(simple_record_strategy())
@settings(max_examples=50)
def test_schema_properties_match_fields(record: SimpleRecord) -> None:
    """Schema properties match dataclass fields."""
    schema_dict = schema(SimpleRecord)
    properties = schema_dict["properties"]
    assert isinstance(properties, dict)
    assert "name" in properties
    assert "count" in properties
    assert "active" in properties
    assert "score" in properties


@given(constrained_record_strategy())
@settings(max_examples=50)
def test_schema_includes_constraints(record: ConstrainedRecord) -> None:
    """Schema includes constraint metadata."""
    schema_dict = schema(ConstrainedRecord)
    properties = schema_dict["properties"]
    assert isinstance(properties, dict)

    # Check text constraints
    text_schema = properties.get("text")
    assert isinstance(text_schema, dict)
    assert text_schema.get("minLength") == 1

    # Check value constraints
    value_schema = properties.get("value")
    assert isinstance(value_schema, dict)
    assert value_schema.get("minimum") == 0
    assert value_schema.get("maximum") == 100


# ============================================================================
# Property Tests: Serialization Options
# ============================================================================


@given(optional_fields_strategy())
@settings(max_examples=100)
def test_exclude_none_option(record: OptionalFieldsRecord) -> None:
    """exclude_none=True excludes None values from output."""
    payload = dump(record, exclude_none=True)
    for key, value in payload.items():
        assert value is not None


@given(simple_record_strategy())
@settings(max_examples=50)
def test_include_dataclass_type(record: SimpleRecord) -> None:
    """include_dataclass_type adds type reference."""
    payload = dump(record, include_dataclass_type=True)
    assert "__type__" in payload
    assert "SimpleRecord" in payload["__type__"]


@given(simple_record_strategy())
@settings(max_examples=50)
def test_custom_type_key(record: SimpleRecord) -> None:
    """Custom type_key is used for type reference."""
    payload = dump(record, include_dataclass_type=True, type_key="$type")
    assert "$type" in payload
    assert "__type__" not in payload


# ============================================================================
# Property Tests: Parse Options
# ============================================================================


@given(simple_record_strategy())
@settings(max_examples=100)
def test_parse_with_extra_ignore(record: SimpleRecord) -> None:
    """extra='ignore' silently ignores extra fields."""
    payload = dump(record)
    payload["extra_field"] = "ignored"
    restored = parse(SimpleRecord, payload, extra="ignore")
    assert restored == record


@given(simple_record_strategy())
@settings(max_examples=50)
def test_parse_with_extra_forbid(record: SimpleRecord) -> None:
    """extra='forbid' raises on extra fields."""
    payload = dump(record)
    payload["extra_field"] = "forbidden"
    with pytest.raises(ValueError) as exc:
        parse(SimpleRecord, payload, extra="forbid")
    assert "extra_field" in str(exc.value).lower()


@given(simple_record_strategy())
@settings(max_examples=100)
def test_parse_case_insensitive(record: SimpleRecord) -> None:
    """case_insensitive=True matches fields regardless of case."""
    payload = {
        "NAME": record.name,
        "COUNT": record.count,
        "ACTIVE": record.active,
        "SCORE": record.score,
    }
    restored = parse(SimpleRecord, payload, case_insensitive=True)
    assert restored == record


# ============================================================================
# Property Tests: Type Coercion
# ============================================================================


@given(st.integers(min_value=0, max_value=1000))
@settings(max_examples=100)
def test_coerce_string_to_int(value: int) -> None:
    """String values are coerced to int when coerce=True."""
    payload = {"name": "test", "count": str(value), "active": True}
    restored = parse(SimpleRecord, payload, coerce=True)
    assert restored.count == value


@given(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False))
@settings(max_examples=100)
def test_coerce_string_to_float(value: float) -> None:
    """String values are coerced to float when coerce=True."""
    payload = {"name": "test", "count": 0, "active": True, "score": str(value)}
    restored = parse(SimpleRecord, payload, coerce=True)
    assert abs(restored.score - value) < 1e-10


@given(st.booleans())
@settings(max_examples=10)
def test_coerce_string_to_bool(value: bool) -> None:
    """String values are coerced to bool when coerce=True."""
    str_value = "true" if value else "false"
    payload = {"name": "test", "count": 0, "active": str_value}
    restored = parse(SimpleRecord, payload, coerce=True)
    assert restored.active == value


# ============================================================================
# Property Tests: Nested Validation Errors
# ============================================================================


def test_nested_constraint_failures_include_paths() -> None:
    """Nested constraint failures include full path in error message."""
    invalid_note = {
        "code": "REC-123",
        "note": {"text": " ", "tags": ["tag-good"]},
        "metrics": {"alpha": 1},
        "optional_notes": [],
    }
    with pytest.raises(ValueError) as note_exc:
        parse(NestedRecord, invalid_note)
    assert "note.text" in str(note_exc.value)

    invalid_metric = {
        "code": "REC-123",
        "note": {"text": "tagged", "tags": ["tag-good"]},
        "metrics": {"alpha": -1},
        "optional_notes": [],
    }
    with pytest.raises(ValueError) as metric_exc:
        parse(NestedRecord, invalid_metric)
    assert "metrics[alpha]" in str(metric_exc.value)

    invalid_rating = {
        "code": "REC-123",
        "note": {"text": "tagged", "tags": ["tag-good"]},
        "metrics": {"alpha": 1},
        "optional_notes": [],
        "rating": 42,
    }
    with pytest.raises(ValueError) as rating_exc:
        parse(NestedRecord, invalid_rating)
    assert "rating" in str(rating_exc.value)


# ============================================================================
# Property Tests: Idempotency
# ============================================================================


@given(simple_record_strategy())
@settings(max_examples=50)
def test_dump_is_idempotent(record: SimpleRecord) -> None:
    """Multiple dump calls produce identical output."""
    payload1 = dump(record)
    payload2 = dump(record)
    assert payload1 == payload2


@given(simple_record_strategy())
@settings(max_examples=50)
def test_double_roundtrip(record: SimpleRecord) -> None:
    """Double roundtrip (dump→parse→dump→parse) preserves data."""
    payload1 = dump(record)
    restored1 = parse(SimpleRecord, payload1)
    payload2 = dump(restored1)
    restored2 = parse(SimpleRecord, payload2)
    assert restored2 == record
    assert payload1 == payload2


# ============================================================================
# Property Tests: Dictionary Data Strategy
# ============================================================================


@given(data=st.dictionaries(
    st.text(alphabet=st.characters(min_codepoint=97, max_codepoint=122), min_size=1, max_size=10),
    st.integers(min_value=0, max_value=100),
    min_size=0,
    max_size=5,
))
@settings(max_examples=100)
def test_dict_field_roundtrip(data: dict[str, int]) -> None:
    """Dict fields with varying content survive roundtrip."""
    @dataclass(frozen=True)
    class DictContainer:
        data: dict[str, int]

    record = DictContainer(data=data)
    payload = dump(record)
    restored = parse(DictContainer, payload)
    assert restored.data == data
