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

"""Session annotation system for dataclass field metadata.

This module provides an annotation system for dataclasses stored in session
snapshots. Annotations reduce UI noise by marking which fields are most
important and how they should be rendered.

Example usage:

    from dataclasses import dataclass, field
    from weakincentives.runtime.annotations import SliceMeta, register_annotations

    @dataclass(slots=True, frozen=True)
    class PlanStep:
        __slice_meta__ = SliceMeta(label="Plan Step")

        step_id: int = field(
            metadata={
                "display": "secondary",
                "description": "Stable identifier for the step.",
            }
        )
        title: str = field(
            metadata={
                "display": "primary",
                "label": "Step",
                "description": "Concise summary of the work item.",
            }
        )

    register_annotations(PlanStep)
"""

from __future__ import annotations

import dataclasses
import threading
from collections.abc import Mapping
from dataclasses import field, fields, is_dataclass
from types import MappingProxyType
from typing import Any, Literal, TypeGuard, cast

from ..dataclasses import FrozenDataclass
from ..serde._utils import type_identifier
from ..types import JSONValue

DisplayLevel = Literal["primary", "secondary", "hidden"]
FormatHint = Literal["text", "markdown", "code", "json"]
SortOrder = Literal["asc", "desc"]


@FrozenDataclass()
class FieldAnnotation:
    """Annotation metadata for a single dataclass field.

    Fields without explicit annotations default to display="secondary" and
    format="text".
    """

    display: DisplayLevel = field(
        default="secondary",
        metadata={"description": "UI prominence level for this field."},
    )
    format: FormatHint = field(
        default="text",
        metadata={"description": "Rendering format hint for this field."},
    )
    label: str | None = field(
        default=None,
        metadata={"description": "Human-readable label (defaults to field name)."},
    )
    description: str | None = field(
        default=None,
        metadata={"description": "Tooltip/help text for this field."},
    )


@FrozenDataclass()
class SliceMeta:
    """Slice-level metadata for dataclasses that participate in session storage.

    Attach as a class attribute named ``__slice_meta__`` on the dataclass.
    """

    label: str = field(
        metadata={"description": "Human-readable name for the slice type."}
    )
    description: str = field(
        default="",
        metadata={"description": "Explanation of what this slice contains."},
    )
    icon: str | None = field(
        default=None,
        metadata={"description": "Icon identifier hint (e.g., Lucide icon name)."},
    )
    sort_key: str | None = field(
        default=None,
        metadata={"description": "Default field to sort instances by."},
    )
    sort_order: SortOrder = field(
        default="asc",
        metadata={"description": "Default sort direction."},
    )


@FrozenDataclass()
class SliceAnnotations:
    """Complete annotation set for a slice type including fields and metadata."""

    type_id: str = field(
        metadata={"description": "Fully qualified type identifier for the slice."}
    )
    slice_meta: SliceMeta | None = field(
        default=None,
        metadata={"description": "Slice-level metadata when defined."},
    )
    fields: Mapping[str, FieldAnnotation] = field(
        default_factory=lambda: cast(
            Mapping[str, FieldAnnotation], MappingProxyType({})
        ),
        metadata={"description": "Field name to annotation mapping."},
    )


# Global registry for annotations
_registry: dict[str, SliceAnnotations] = {}
_registry_lock = threading.Lock()


def _is_dataclass_type(value: object) -> TypeGuard[type[Any]]:
    """Check if value is a dataclass type (not an instance)."""
    return isinstance(value, type) and is_dataclass(value)


_VALID_DISPLAY_LEVELS: frozenset[str] = frozenset({"primary", "secondary", "hidden"})
_VALID_FORMAT_HINTS: frozenset[str] = frozenset({"text", "markdown", "code", "json"})
_VALID_SORT_ORDERS: frozenset[str] = frozenset({"asc", "desc"})


def _extract_field_annotation(field_def: dataclasses.Field[object]) -> FieldAnnotation:
    """Extract annotation from a dataclass field's metadata."""
    metadata = field_def.metadata

    display = metadata.get("display", "secondary")
    if display not in _VALID_DISPLAY_LEVELS:
        display = "secondary"

    format_hint = metadata.get("format", "text")
    if format_hint not in _VALID_FORMAT_HINTS:
        format_hint = "text"

    label = metadata.get("label")
    if label is not None and not isinstance(label, str):
        label = None

    description = metadata.get("description")
    if description is not None and not isinstance(description, str):
        description = None

    return FieldAnnotation(
        display=cast(DisplayLevel, display),
        format=cast(FormatHint, format_hint),
        label=label,
        description=description,
    )


def _extract_slice_meta(cls: type[object]) -> SliceMeta | None:
    """Extract slice metadata from a dataclass class attribute."""
    slice_meta = getattr(cls, "__slice_meta__", None)
    if slice_meta is None:
        return None
    if isinstance(slice_meta, SliceMeta):
        return slice_meta
    return None


def register_annotations(cls: type[object]) -> None:
    """Register annotations for a dataclass type.

    Extracts field annotations from ``field.metadata`` and slice metadata from
    the ``__slice_meta__`` class attribute if present.

    Args:
        cls: A dataclass type to register.

    Raises:
        TypeError: If cls is not a dataclass type.
    """
    if not _is_dataclass_type(cls):
        msg = f"register_annotations requires a dataclass type, got {type(cls)}"
        raise TypeError(msg)

    type_id = type_identifier(cls)
    field_annotations: dict[str, FieldAnnotation] = {}

    for field_def in fields(cls):
        field_annotations[field_def.name] = _extract_field_annotation(field_def)

    slice_meta = _extract_slice_meta(cls)

    annotations = SliceAnnotations(
        type_id=type_id,
        slice_meta=slice_meta,
        fields=MappingProxyType(field_annotations),
    )

    with _registry_lock:
        _registry[type_id] = annotations


def get_field_annotations(cls: type[object]) -> Mapping[str, FieldAnnotation]:
    """Return field annotations for a registered dataclass type.

    Args:
        cls: A dataclass type to look up.

    Returns:
        Mapping from field name to annotation. Returns empty mapping if
        the type is not registered.
    """
    if not _is_dataclass_type(cls):
        return MappingProxyType({})

    type_id = type_identifier(cls)
    with _registry_lock:
        annotations = _registry.get(type_id)

    if annotations is None:
        return MappingProxyType({})

    return annotations.fields


def get_slice_meta(cls: type[object]) -> SliceMeta | None:
    """Return slice metadata for a registered dataclass type.

    Args:
        cls: A dataclass type to look up.

    Returns:
        SliceMeta if the type has slice metadata registered, None otherwise.
    """
    if not _is_dataclass_type(cls):
        return None

    type_id = type_identifier(cls)
    with _registry_lock:
        annotations = _registry.get(type_id)

    if annotations is None:
        return None

    return annotations.slice_meta


def get_all_registered() -> Mapping[str, SliceAnnotations]:
    """Return all registered slice annotations.

    Returns:
        Immutable mapping from type identifier to annotations.
    """
    with _registry_lock:
        return MappingProxyType(dict(_registry))


def get_annotations_for_type_id(type_id: str) -> SliceAnnotations | None:
    """Return annotations for a type by its identifier string.

    Args:
        type_id: Fully qualified type identifier.

    Returns:
        SliceAnnotations if registered, None otherwise.
    """
    with _registry_lock:
        return _registry.get(type_id)


def clear_registry() -> None:
    """Clear all registered annotations. Primarily for testing."""
    with _registry_lock:
        _registry.clear()


def build_header(type_ids: set[str]) -> Mapping[str, JSONValue]:
    """Build a JSONL header from a set of type identifiers.

    The header contains annotation metadata for the specified types,
    making snapshot files self-describing.

    Args:
        type_ids: Set of type identifiers to include in the header.

    Returns:
        Header payload suitable for JSON serialization.
    """
    slices: dict[str, JSONValue] = {}

    with _registry_lock:
        for type_id in sorted(type_ids):
            annotations = _registry.get(type_id)
            if annotations is None:
                continue

            slice_data: dict[str, JSONValue] = {}

            # Add slice metadata if present
            if annotations.slice_meta is not None:
                meta = annotations.slice_meta
                slice_data["label"] = meta.label
                if meta.description:
                    slice_data["description"] = meta.description
                if meta.icon is not None:
                    slice_data["icon"] = meta.icon
                if meta.sort_key is not None:
                    slice_data["sort_key"] = meta.sort_key
                slice_data["sort_order"] = meta.sort_order

            # Add field annotations
            fields_data: dict[str, JSONValue] = {}
            for field_name, field_ann in annotations.fields.items():
                field_data: dict[str, JSONValue] = {
                    "display": field_ann.display,
                    "format": field_ann.format,
                }
                if field_ann.label is not None:
                    field_data["label"] = field_ann.label
                if field_ann.description is not None:
                    field_data["description"] = field_ann.description
                fields_data[field_name] = field_data

            slice_data["fields"] = fields_data
            slices[type_id] = slice_data

    return {
        "header": True,
        "annotation_version": "1",
        "slices": slices,
    }


def _parse_slice_meta_from_data(
    slice_data: Mapping[str, JSONValue],
) -> SliceMeta | None:
    """Parse slice metadata from JSON data."""
    label = slice_data.get("label")
    if not isinstance(label, str):
        return None

    description = slice_data.get("description", "")
    if not isinstance(description, str):
        description = ""

    icon = slice_data.get("icon")
    if icon is not None and not isinstance(icon, str):
        icon = None

    sort_key = slice_data.get("sort_key")
    if sort_key is not None and not isinstance(sort_key, str):
        sort_key = None

    sort_order = slice_data.get("sort_order", "asc")
    if sort_order not in _VALID_SORT_ORDERS:
        sort_order = "asc"

    return SliceMeta(
        label=label,
        description=description,
        icon=icon,
        sort_key=sort_key,
        sort_order=cast(SortOrder, sort_order),
    )


def _parse_field_annotation_from_data(
    field_data: Mapping[str, JSONValue],
) -> FieldAnnotation:
    """Parse a single field annotation from JSON data."""
    display = field_data.get("display", "secondary")
    if display not in _VALID_DISPLAY_LEVELS:
        display = "secondary"

    format_hint = field_data.get("format", "text")
    if format_hint not in _VALID_FORMAT_HINTS:
        format_hint = "text"

    label = field_data.get("label")
    if label is not None and not isinstance(label, str):
        label = None

    description = field_data.get("description")
    if description is not None and not isinstance(description, str):
        description = None

    return FieldAnnotation(
        display=cast(DisplayLevel, display),
        format=cast(FormatHint, format_hint),
        label=label,
        description=description,
    )


def _parse_fields_from_data(
    slice_data: Mapping[str, JSONValue],
) -> dict[str, FieldAnnotation]:
    """Parse field annotations from slice JSON data."""
    field_annotations: dict[str, FieldAnnotation] = {}
    fields_data = slice_data.get("fields")
    if not isinstance(fields_data, Mapping):
        return field_annotations

    # Cast the mapping to known types after isinstance check
    typed_fields: Mapping[str, JSONValue] = cast(Mapping[str, JSONValue], fields_data)
    for field_name, field_data in typed_fields.items():
        if not isinstance(field_data, Mapping):
            continue
        typed_field_data = cast(Mapping[str, JSONValue], field_data)
        field_annotations[field_name] = _parse_field_annotation_from_data(
            typed_field_data
        )

    return field_annotations


def parse_header(payload: Mapping[str, JSONValue]) -> Mapping[str, SliceAnnotations]:
    """Parse a JSONL header into SliceAnnotations.

    Args:
        payload: Parsed JSON header object.

    Returns:
        Mapping from type identifier to annotations.
    """
    if not payload.get("header"):
        return MappingProxyType({})

    slices_data = payload.get("slices")
    if not isinstance(slices_data, Mapping):
        return MappingProxyType({})

    result: dict[str, SliceAnnotations] = {}
    # Cast the mapping to known types after isinstance check
    typed_slices: Mapping[str, JSONValue] = cast(Mapping[str, JSONValue], slices_data)
    for type_id, slice_data in typed_slices.items():
        if not isinstance(slice_data, Mapping):
            continue

        typed_slice_data = cast(Mapping[str, JSONValue], slice_data)
        slice_meta = _parse_slice_meta_from_data(typed_slice_data)
        field_annotations = _parse_fields_from_data(typed_slice_data)

        result[type_id] = SliceAnnotations(
            type_id=type_id,
            slice_meta=slice_meta,
            fields=MappingProxyType(field_annotations),
        )

    return MappingProxyType(result)


def is_header_line(line: str) -> bool:
    """Check if a JSONL line is a header line.

    Args:
        line: A stripped JSONL line.

    Returns:
        True if the line contains the header marker.
    """
    # Check for header marker in various positions (keys may be sorted)
    return (
        '"header": true' in line
        or '"header":true' in line
        or line.startswith('{"header":')
        or line.startswith('{ "header":')
    )


__all__ = [
    "DisplayLevel",
    "FieldAnnotation",
    "FormatHint",
    "SliceAnnotations",
    "SliceMeta",
    "SortOrder",
    "build_header",
    "clear_registry",
    "get_all_registered",
    "get_annotations_for_type_id",
    "get_field_annotations",
    "get_slice_meta",
    "is_header_line",
    "parse_header",
    "register_annotations",
]
