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

"""Snapshot serialization utilities for :mod:`weakincentives.runtime.session`."""

from __future__ import annotations

import json
import types
from collections.abc import Mapping
from dataclasses import field, is_dataclass
from datetime import UTC, datetime
from typing import TypeGuard, cast, override
from uuid import UUID

from ...dataclasses import FrozenDataclass
from ...errors import WinkError
from ...prompt._types import SupportsDataclass
from ...serde import dump, parse
from ...serde._utils import (
    TYPE_REF_KEY,
    resolve_type_identifier,
    type_identifier,
)
from ...types import JSONValue
from ._slice_types import SessionSlice, SessionSliceType
from .dataclasses import is_dataclass_instance

SNAPSHOT_SCHEMA_VERSION = "1"


type SnapshotState = Mapping[SessionSliceType, SessionSlice]


class SnapshotSerializationError(WinkError, RuntimeError):
    """Raised when snapshot capture fails due to unsupported payloads."""


class SnapshotRestoreError(WinkError, RuntimeError):
    """Raised when snapshot restoration fails due to incompatible payloads."""


def _normalize_tags(
    tags: Mapping[object, object] | None, *, error_cls: type[Exception]
) -> Mapping[str, str]:
    normalized: dict[str, str] = {}

    if tags is not None:
        for key, value in tags.items():
            if not isinstance(key, str) or not isinstance(value, str):
                msg = "Snapshot tags must be string key/value pairs"
                raise error_cls(msg)
            normalized[key] = value

    return cast(Mapping[str, str], types.MappingProxyType(normalized))


def normalize_snapshot_state(
    state: Mapping[SessionSliceType, SessionSlice],
) -> SnapshotState:
    """Validate snapshot state and return an immutable copy."""

    normalized: dict[SessionSliceType, SessionSlice] = {}
    for slice_key, values in state.items():
        if not _is_dataclass_type(slice_key):
            raise ValueError("Slice keys must be dataclass types")

        slice_type = slice_key

        items: list[SupportsDataclass] = []
        for value in values:
            if not is_dataclass_instance(value):
                raise ValueError(
                    f"Slice {slice_type.__qualname__} contains non-dataclass value"
                )
            try:
                _ = dump(value)
            except Exception as error:
                raise ValueError(
                    f"Slice {slice_type.__qualname__} cannot be serialized"
                ) from error
            items.append(value)

        normalized[slice_type] = tuple(items)

    return cast(SnapshotState, types.MappingProxyType(normalized))


def _type_identifier(cls: SessionSliceType) -> str:
    return type_identifier(cls)


def _resolve_type(identifier: str) -> SessionSliceType:
    try:
        return cast(SessionSliceType, resolve_type_identifier(identifier))
    except (TypeError, ValueError) as error:
        raise SnapshotRestoreError(str(error)) from error


def _ensure_timezone(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt


def _load_snapshot_object(raw: str) -> Mapping[str, JSONValue]:
    try:
        payload_obj: JSONValue = json.loads(raw)
    except json.JSONDecodeError as error:
        raise SnapshotRestoreError("Invalid snapshot JSON") from error

    if not isinstance(payload_obj, Mapping):
        raise SnapshotRestoreError("Snapshot payload must be an object")

    return cast(Mapping[str, JSONValue], payload_obj)


def _validate_schema_version(version: str) -> None:
    if version != SNAPSHOT_SCHEMA_VERSION:
        msg = (
            "Snapshot schema version mismatch: "
            f"expected {SNAPSHOT_SCHEMA_VERSION}, got {version!r}"
        )
        raise SnapshotRestoreError(msg)


def _validate_payload_fields(
    payload: Mapping[str, JSONValue],
) -> tuple[str, str, str | None]:
    version_obj = payload.get("version")
    if not isinstance(version_obj, str):
        raise SnapshotRestoreError("Snapshot version must be a string")

    created_at_obj = payload.get("created_at")
    if not isinstance(created_at_obj, str):
        raise SnapshotRestoreError("Snapshot created_at must be a string")

    parent_id_obj = payload.get("parent_id")
    if parent_id_obj is not None and not isinstance(parent_id_obj, str):
        raise SnapshotRestoreError("Snapshot parent_id must be a string")

    return version_obj, created_at_obj, parent_id_obj


def _validate_children_ids(payload: Mapping[str, JSONValue]) -> tuple[str, ...]:
    children_ids_obj = payload.get("children_ids", [])
    if not isinstance(children_ids_obj, list):
        raise SnapshotRestoreError("Snapshot children_ids must be a list")

    children_ids: list[str] = []
    for child_id in children_ids_obj:
        if not isinstance(child_id, str):
            raise SnapshotRestoreError("Snapshot children_ids entries must be strings")
        children_ids.append(child_id)

    return tuple(children_ids)


def _validate_slices(
    payload: Mapping[str, JSONValue],
) -> tuple[SnapshotSlicePayload, ...]:
    slices_obj = payload.get("slices", [])
    if not isinstance(slices_obj, list):
        raise SnapshotRestoreError("Snapshot slices must be a list")

    slices_source = slices_obj
    return tuple(SnapshotSlicePayload.from_object(entry) for entry in slices_source)


def _validate_tags(payload: Mapping[str, JSONValue]) -> Mapping[str, str]:
    tags_obj = payload.get("tags", {})
    if not isinstance(tags_obj, Mapping):
        raise SnapshotRestoreError("Snapshot tags must be an object")

    return _normalize_tags(
        cast(Mapping[object, object] | None, tags_obj),
        error_cls=SnapshotRestoreError,
    )


def _construct_snapshot_payload(
    cls: type[SnapshotPayload],
    payload: Mapping[str, JSONValue],
) -> SnapshotPayload:
    version, created_at, parent_id = _validate_payload_fields(payload)
    children_ids = _validate_children_ids(payload)
    slices = _validate_slices(payload)
    tags = _validate_tags(payload)

    return cls(
        version=version,
        created_at=created_at,
        slices=slices,
        parent_id=parent_id,
        children_ids=children_ids,
        tags=tags,
    )


def _infer_item_type(
    slice_type: SessionSliceType, values: SessionSlice
) -> SessionSliceType:
    if values:
        first_value = values[0]
        first_type = type(first_value)
        for value in values:
            if type(value) is not first_type:  # intentional identity check
                msg = (
                    "Snapshot slices must contain a single dataclass type; "
                    f"found {type(value)!r}"
                )
                raise SnapshotSerializationError(msg)
        return first_type
    return slice_type


def _is_dataclass_type(value: object) -> TypeGuard[type[SupportsDataclass]]:
    return isinstance(value, type) and is_dataclass(value)


@FrozenDataclass()
class SnapshotSlicePayload:
    """Typed representation of a serialized snapshot slice entry."""

    slice_type: str
    item_type: str
    items: tuple[Mapping[str, JSONValue], ...]

    @classmethod
    def from_object(cls, obj: object) -> SnapshotSlicePayload:
        if not isinstance(obj, Mapping):
            raise SnapshotRestoreError("Slice entry must be an object")

        entry = cast(Mapping[str, JSONValue], obj)
        slice_identifier = entry.get("slice_type")
        item_identifier = entry.get("item_type")

        if not isinstance(slice_identifier, str) or not isinstance(
            item_identifier, str
        ):
            raise SnapshotRestoreError("Slice type identifiers must be strings")

        items_obj_raw = entry.get("items", [])
        if not isinstance(items_obj_raw, list):
            raise SnapshotRestoreError("Slice items must be a list")

        items_obj = items_obj_raw
        items: list[Mapping[str, JSONValue]] = []
        for item in items_obj:
            if not isinstance(item, Mapping):
                raise SnapshotRestoreError("Slice items must be objects")
            items.append(cast(Mapping[str, JSONValue], item))

        return cls(
            slice_type=slice_identifier,
            item_type=item_identifier,
            items=tuple(items),
        )


@FrozenDataclass()
class SnapshotPayload:
    """Typed representation of the serialized snapshot envelope."""

    version: str
    created_at: str
    slices: tuple[SnapshotSlicePayload, ...]
    parent_id: str | None = None
    children_ids: tuple[str, ...] = ()
    tags: Mapping[str, str] = field(
        default_factory=lambda: cast(Mapping[str, str], types.MappingProxyType({}))
    )

    @classmethod
    def from_json(cls, raw: str) -> SnapshotPayload:
        payload = _load_snapshot_object(raw)
        return _construct_snapshot_payload(cls, payload)


@FrozenDataclass()
class Snapshot:
    """Frozen value object representing session slice state."""

    created_at: datetime
    parent_id: UUID | None = None
    children_ids: tuple[UUID, ...] = ()
    slices: SnapshotState = field(
        default_factory=lambda: cast(
            SnapshotState,
            types.MappingProxyType({}),
        )
    )
    tags: Mapping[str, str] = field(
        default_factory=lambda: cast(Mapping[str, str], types.MappingProxyType({}))
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "created_at", _ensure_timezone(self.created_at))
        object.__setattr__(
            self,
            "slices",
            cast(SnapshotState, types.MappingProxyType(dict(self.slices))),
        )
        object.__setattr__(
            self,
            "tags",
            cast(Mapping[str, str], types.MappingProxyType(dict(self.tags))),
        )

    @override
    def __hash__(self) -> int:
        ordered = tuple(
            sorted(
                self.slices.items(),
                key=lambda item: _type_identifier(item[0]),
            )
        )
        ordered_tags = tuple(sorted(self.tags.items()))
        return hash(
            (self.created_at, ordered, self.parent_id, self.children_ids, ordered_tags)
        )

    def to_json(self) -> str:
        """Serialize the snapshot to a JSON string."""

        payload_slices: list[dict[str, JSONValue]] = []
        for slice_type, values in sorted(
            self.slices.items(), key=lambda item: _type_identifier(item[0])
        ):
            item_type = _infer_item_type(slice_type, values)
            try:
                serialized_items = [
                    cast(
                        Mapping[str, JSONValue],
                        dump(
                            value,
                            include_dataclass_type=True,
                            type_key=TYPE_REF_KEY,
                        ),
                    )
                    for value in values
                ]
            except Exception as error:
                msg = f"Failed to serialize slice {slice_type.__qualname__}"
                raise SnapshotSerializationError(msg) from error

            payload_slices.append(
                {
                    "slice_type": _type_identifier(slice_type),
                    "item_type": _type_identifier(item_type),
                    "items": serialized_items,
                }
            )

        payload: dict[str, JSONValue] = {
            "version": SNAPSHOT_SCHEMA_VERSION,
            "created_at": self.created_at.isoformat(),
            "parent_id": str(self.parent_id) if self.parent_id is not None else None,
            "children_ids": [str(child) for child in self.children_ids],
            "slices": payload_slices,
            "tags": dict(sorted(self.tags.items())),
        }
        return json.dumps(payload, sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> Snapshot:
        """Deserialize a snapshot from its JSON representation."""

        payload = SnapshotPayload.from_json(raw)

        _validate_schema_version(payload.version)

        try:
            created_at = datetime.fromisoformat(payload.created_at)
        except ValueError as error:
            raise SnapshotRestoreError("Invalid created_at timestamp") from error

        try:
            parent_id = (
                UUID(payload.parent_id) if payload.parent_id is not None else None
            )
        except ValueError as error:
            raise SnapshotRestoreError("Invalid parent_id") from error

        try:
            children_ids = tuple(UUID(value) for value in payload.children_ids)
        except ValueError as error:
            raise SnapshotRestoreError("Invalid children_ids entry") from error

        restored: dict[SessionSliceType, SessionSlice] = {}
        for entry in payload.slices:
            slice_type_candidate = _resolve_type(entry.slice_type)
            item_type_candidate = _resolve_type(entry.item_type)

            if not _is_dataclass_type(slice_type_candidate) or not _is_dataclass_type(
                item_type_candidate
            ):
                raise SnapshotRestoreError("Snapshot types must be dataclasses")

            slice_type = slice_type_candidate
            item_type = item_type_candidate

            restored_items: list[SupportsDataclass] = []
            for item_mapping in entry.items:
                try:
                    restored_item = parse(
                        item_type,
                        item_mapping,
                        allow_dataclass_type=True,
                        type_key=TYPE_REF_KEY,
                    )
                except Exception as error:
                    raise SnapshotRestoreError(
                        f"Failed to restore slice {slice_type.__qualname__}"
                    ) from error
                else:
                    restored_items.append(restored_item)

            restored[slice_type] = tuple(restored_items)

        return cls(
            created_at=_ensure_timezone(created_at),
            parent_id=parent_id,
            children_ids=children_ids,
            slices=restored,
            tags=payload.tags,
        )
