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

"""Tests for session annotation system."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import MappingProxyType

import pytest

from weakincentives.dataclasses import FrozenDataclass
from weakincentives.runtime.annotations import (
    FieldAnnotation,
    SliceAnnotations,
    SliceMeta,
    build_header,
    clear_registry,
    get_all_registered,
    get_annotations_for_type_id,
    get_field_annotations,
    get_slice_meta,
    is_header_line,
    parse_header,
    register_annotations,
)


@pytest.fixture(autouse=True)
def _clear_registry_fixture() -> None:
    """Clear the annotation registry before each test."""
    clear_registry()


class TestFieldAnnotation:
    def test_default_values(self) -> None:
        ann = FieldAnnotation()
        assert ann.display == "secondary"
        assert ann.format == "text"
        assert ann.label is None
        assert ann.description is None

    def test_custom_values(self) -> None:
        ann = FieldAnnotation(
            display="primary",
            format="markdown",
            label="Title",
            description="A description",
        )
        assert ann.display == "primary"
        assert ann.format == "markdown"
        assert ann.label == "Title"
        assert ann.description == "A description"


class TestSliceMeta:
    def test_required_label(self) -> None:
        meta = SliceMeta(label="Test Slice")
        assert meta.label == "Test Slice"
        assert meta.description == ""
        assert meta.icon is None
        assert meta.sort_key is None
        assert meta.sort_order == "asc"

    def test_all_fields(self) -> None:
        meta = SliceMeta(
            label="Task",
            description="A task item",
            icon="check",
            sort_key="created_at",
            sort_order="desc",
        )
        assert meta.label == "Task"
        assert meta.description == "A task item"
        assert meta.icon == "check"
        assert meta.sort_key == "created_at"
        assert meta.sort_order == "desc"


class TestSliceAnnotations:
    def test_basic_construction(self) -> None:
        ann = SliceAnnotations(type_id="test.MyClass")
        assert ann.type_id == "test.MyClass"
        assert ann.slice_meta is None
        assert ann.fields == MappingProxyType({})

    def test_with_fields(self) -> None:
        fields_data = {"name": FieldAnnotation(display="primary")}
        ann = SliceAnnotations(
            type_id="test.MyClass",
            fields=MappingProxyType(fields_data),
        )
        assert "name" in ann.fields
        assert ann.fields["name"].display == "primary"


class TestRegisterAnnotations:
    def test_register_simple_dataclass(self) -> None:
        @dataclass(slots=True, frozen=True)
        class SimpleClass:
            name: str
            count: int = 0

        register_annotations(SimpleClass)
        annotations = get_field_annotations(SimpleClass)

        assert "name" in annotations
        assert "count" in annotations
        assert annotations["name"].display == "secondary"  # default

    def test_register_with_metadata(self) -> None:
        @dataclass(slots=True, frozen=True)
        class AnnotatedClass:
            title: str = field(
                metadata={
                    "display": "primary",
                    "format": "text",
                    "label": "Title",
                    "description": "The title field",
                }
            )

        register_annotations(AnnotatedClass)
        annotations = get_field_annotations(AnnotatedClass)

        title_ann = annotations["title"]
        assert title_ann.display == "primary"
        assert title_ann.format == "text"
        assert title_ann.label == "Title"
        assert title_ann.description == "The title field"

    def test_register_with_slice_meta(self) -> None:
        @FrozenDataclass()
        class SliceWithMeta:
            __slice_meta__ = SliceMeta(
                label="Test Slice",
                description="Test description",
                icon="star",
            )

            value: str

        register_annotations(SliceWithMeta)
        meta = get_slice_meta(SliceWithMeta)

        assert meta is not None
        assert meta.label == "Test Slice"
        assert meta.description == "Test description"
        assert meta.icon == "star"

    def test_register_non_dataclass_raises(self) -> None:
        class NotADataclass:
            pass

        with pytest.raises(TypeError, match="requires a dataclass type"):
            register_annotations(NotADataclass)

    def test_invalid_display_defaults_to_secondary(self) -> None:
        @dataclass(slots=True, frozen=True)
        class BadDisplay:
            value: str = field(metadata={"display": "invalid"})

        register_annotations(BadDisplay)
        annotations = get_field_annotations(BadDisplay)
        assert annotations["value"].display == "secondary"

    def test_invalid_format_defaults_to_text(self) -> None:
        @dataclass(slots=True, frozen=True)
        class BadFormat:
            value: str = field(metadata={"format": "invalid"})

        register_annotations(BadFormat)
        annotations = get_field_annotations(BadFormat)
        assert annotations["value"].format == "text"

    def test_invalid_label_type_defaults_to_none(self) -> None:
        @dataclass(slots=True, frozen=True)
        class BadLabel:
            value: str = field(metadata={"label": 123})

        register_annotations(BadLabel)
        annotations = get_field_annotations(BadLabel)
        assert annotations["value"].label is None

    def test_invalid_description_type_defaults_to_none(self) -> None:
        @dataclass(slots=True, frozen=True)
        class BadDescription:
            value: str = field(metadata={"description": ["list"]})

        register_annotations(BadDescription)
        annotations = get_field_annotations(BadDescription)
        assert annotations["value"].description is None

    def test_invalid_slice_meta_type_ignored(self) -> None:
        @dataclass(slots=True, frozen=True)
        class BadSliceMeta:
            __slice_meta__ = "not a SliceMeta"
            value: str

        register_annotations(BadSliceMeta)
        assert get_slice_meta(BadSliceMeta) is None


class TestRegistryLookup:
    def test_get_unregistered_returns_empty(self) -> None:
        @dataclass(slots=True, frozen=True)
        class Unregistered:
            value: str

        assert get_field_annotations(Unregistered) == MappingProxyType({})
        assert get_slice_meta(Unregistered) is None

    def test_get_non_dataclass_returns_empty(self) -> None:
        class NotDataclass:
            pass

        assert get_field_annotations(NotDataclass) == MappingProxyType({})
        assert get_slice_meta(NotDataclass) is None

    def test_get_all_registered(self) -> None:
        @dataclass(slots=True, frozen=True)
        class First:
            value: str

        @dataclass(slots=True, frozen=True)
        class Second:
            count: int

        register_annotations(First)
        register_annotations(Second)

        all_registered = get_all_registered()
        assert len(all_registered) == 2

    def test_get_annotations_for_type_id(self) -> None:
        @dataclass(slots=True, frozen=True)
        class TypedClass:
            value: str = field(metadata={"display": "primary"})

        register_annotations(TypedClass)
        type_id = f"{TypedClass.__module__}:{TypedClass.__qualname__}"

        ann = get_annotations_for_type_id(type_id)
        assert ann is not None
        assert ann.type_id == type_id

    def test_get_annotations_unknown_type_id(self) -> None:
        assert get_annotations_for_type_id("unknown.Type") is None


class TestBuildHeader:
    def test_build_empty_header(self) -> None:
        header = build_header(set())
        assert header["header"] is True
        assert header["annotation_version"] == "1"
        assert header["slices"] == {}

    def test_build_header_with_registered_types(self) -> None:
        @FrozenDataclass()
        class TestSlice:
            __slice_meta__ = SliceMeta(label="Test", icon="star")

            name: str = field(
                metadata={
                    "display": "primary",
                    "format": "text",
                    "label": "Name",
                    "description": "The name field",
                }
            )

        register_annotations(TestSlice)
        # type_identifier uses module:qualname format
        type_id = f"{TestSlice.__module__}:{TestSlice.__qualname__}"

        header = build_header({type_id})

        slices = header["slices"]
        assert isinstance(slices, dict)
        assert type_id in slices
        slice_data = slices[type_id]
        assert isinstance(slice_data, dict)
        assert slice_data["label"] == "Test"
        assert slice_data["icon"] == "star"
        assert "fields" in slice_data
        fields = slice_data["fields"]
        assert isinstance(fields, dict)
        name_field = fields["name"]
        assert isinstance(name_field, dict)
        assert name_field["display"] == "primary"

    def test_build_header_unregistered_type_ignored(self) -> None:
        header = build_header({"unknown.Type"})
        assert header["slices"] == {}


class TestParseHeader:
    def test_parse_empty_header(self) -> None:
        payload = {"header": True, "annotation_version": "1", "slices": {}}
        result = parse_header(payload)
        assert result == MappingProxyType({})

    def test_parse_non_header(self) -> None:
        payload = {"something": "else"}
        result = parse_header(payload)
        assert result == MappingProxyType({})

    def test_parse_header_with_annotations(self) -> None:
        payload = {
            "header": True,
            "annotation_version": "1",
            "slices": {
                "test.MyClass": {
                    "label": "My Class",
                    "description": "A test class",
                    "icon": "box",
                    "sort_key": "created_at",
                    "sort_order": "desc",
                    "fields": {
                        "name": {
                            "display": "primary",
                            "format": "text",
                            "label": "Name",
                            "description": "The name",
                        },
                        "content": {
                            "display": "secondary",
                            "format": "markdown",
                        },
                    },
                }
            },
        }

        result = parse_header(payload)

        assert "test.MyClass" in result
        ann = result["test.MyClass"]
        assert ann.slice_meta is not None
        assert ann.slice_meta.label == "My Class"
        assert ann.slice_meta.description == "A test class"
        assert ann.slice_meta.icon == "box"
        assert ann.slice_meta.sort_key == "created_at"
        assert ann.slice_meta.sort_order == "desc"

        assert ann.fields["name"].display == "primary"
        assert ann.fields["name"].format == "text"
        assert ann.fields["name"].label == "Name"
        assert ann.fields["name"].description == "The name"

        assert ann.fields["content"].display == "secondary"
        assert ann.fields["content"].format == "markdown"

    def test_parse_header_invalid_values_use_defaults(self) -> None:
        payload = {
            "header": True,
            "annotation_version": "1",
            "slices": {
                "test.Type": {
                    "label": "Type",
                    "fields": {
                        "field": {
                            "display": "invalid",
                            "format": "invalid",
                        }
                    },
                }
            },
        }

        result = parse_header(payload)
        field_ann = result["test.Type"].fields["field"]
        assert field_ann.display == "secondary"
        assert field_ann.format == "text"

    def test_parse_header_invalid_description_type(self) -> None:
        payload = {
            "header": True,
            "annotation_version": "1",
            "slices": {
                "test.Type": {
                    "label": "Type",
                    "description": 123,  # Invalid type
                }
            },
        }
        result = parse_header(payload)
        assert result["test.Type"].slice_meta is not None
        assert result["test.Type"].slice_meta.description == ""

    def test_parse_header_invalid_icon_type(self) -> None:
        payload = {
            "header": True,
            "annotation_version": "1",
            "slices": {
                "test.Type": {
                    "label": "Type",
                    "icon": 123,  # Invalid type
                }
            },
        }
        result = parse_header(payload)
        assert result["test.Type"].slice_meta is not None
        assert result["test.Type"].slice_meta.icon is None

    def test_parse_header_invalid_sort_key_type(self) -> None:
        payload = {
            "header": True,
            "annotation_version": "1",
            "slices": {
                "test.Type": {
                    "label": "Type",
                    "sort_key": ["list"],  # Invalid type
                }
            },
        }
        result = parse_header(payload)
        assert result["test.Type"].slice_meta is not None
        assert result["test.Type"].slice_meta.sort_key is None

    def test_parse_header_invalid_sort_order(self) -> None:
        payload = {
            "header": True,
            "annotation_version": "1",
            "slices": {
                "test.Type": {
                    "label": "Type",
                    "sort_order": "invalid",
                }
            },
        }
        result = parse_header(payload)
        assert result["test.Type"].slice_meta is not None
        assert result["test.Type"].slice_meta.sort_order == "asc"

    def test_parse_header_invalid_field_label_type(self) -> None:
        payload = {
            "header": True,
            "annotation_version": "1",
            "slices": {
                "test.Type": {
                    "fields": {
                        "field": {
                            "label": 123,  # Invalid type
                        }
                    },
                }
            },
        }
        result = parse_header(payload)
        field_ann = result["test.Type"].fields["field"]
        assert field_ann.label is None

    def test_parse_header_invalid_field_description_type(self) -> None:
        payload = {
            "header": True,
            "annotation_version": "1",
            "slices": {
                "test.Type": {
                    "fields": {
                        "field": {
                            "description": {"dict": "value"},  # Invalid type
                        }
                    },
                }
            },
        }
        result = parse_header(payload)
        field_ann = result["test.Type"].fields["field"]
        assert field_ann.description is None

    def test_parse_header_non_mapping_fields(self) -> None:
        payload = {
            "header": True,
            "annotation_version": "1",
            "slices": {
                "test.Type": {
                    "fields": "not a mapping",
                }
            },
        }
        result = parse_header(payload)
        assert result["test.Type"].fields == MappingProxyType({})

    def test_parse_header_non_mapping_field_data(self) -> None:
        payload = {
            "header": True,
            "annotation_version": "1",
            "slices": {
                "test.Type": {
                    "fields": {
                        "field": "not a mapping",
                    }
                },
            },
        }
        result = parse_header(payload)
        assert "field" not in result["test.Type"].fields

    def test_parse_header_non_mapping_slice_data(self) -> None:
        payload = {
            "header": True,
            "annotation_version": "1",
            "slices": {
                "test.Type": "not a mapping",
            },
        }
        result = parse_header(payload)
        assert "test.Type" not in result

    def test_parse_header_slices_not_mapping(self) -> None:
        """Test when slices value itself is not a mapping."""
        payload = {"header": True, "slices": "not a mapping"}
        result = parse_header(payload)
        assert result == MappingProxyType({})


class TestIsHeaderLine:
    def test_header_line_detected(self) -> None:
        line = '{"header": true, "annotation_version": "1", "slices": {}}'
        assert is_header_line(line)

    def test_header_line_with_space(self) -> None:
        line = '{ "header": true }'
        assert is_header_line(line)

    def test_non_header_line(self) -> None:
        line = '{"version": "1", "slices": {}}'
        assert not is_header_line(line)

    def test_snapshot_line_not_header(self) -> None:
        line = '{"version": "1", "created_at": "2024-01-01", "slices": []}'
        assert not is_header_line(line)


class TestRoundTrip:
    def test_build_and_parse_roundtrip(self) -> None:
        @FrozenDataclass()
        class RoundTripSlice:
            __slice_meta__ = SliceMeta(
                label="Round Trip",
                description="Test roundtrip",
                icon="rotate",
                sort_key="id",
                sort_order="asc",
            )

            id: int = field(
                metadata={
                    "display": "secondary",
                    "description": "Identifier",
                }
            )
            body: str = field(
                metadata={
                    "display": "primary",
                    "format": "markdown",
                    "label": "Content",
                    "description": "The body content",
                }
            )

        register_annotations(RoundTripSlice)
        type_id = f"{RoundTripSlice.__module__}:{RoundTripSlice.__qualname__}"

        # Build header
        header = build_header({type_id})

        # Serialize and deserialize (simulates JSONL write/read)
        json_str = json.dumps(header)
        parsed_data = json.loads(json_str)

        # Parse header
        result = parse_header(parsed_data)

        assert type_id in result
        ann = result[type_id]

        # Verify slice meta
        assert ann.slice_meta is not None
        assert ann.slice_meta.label == "Round Trip"
        assert ann.slice_meta.description == "Test roundtrip"
        assert ann.slice_meta.icon == "rotate"
        assert ann.slice_meta.sort_key == "id"
        assert ann.slice_meta.sort_order == "asc"

        # Verify fields
        assert ann.fields["id"].display == "secondary"
        assert ann.fields["id"].description == "Identifier"
        assert ann.fields["body"].display == "primary"
        assert ann.fields["body"].format == "markdown"
        assert ann.fields["body"].label == "Content"
        assert ann.fields["body"].description == "The body content"
