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

"""Snapshot serialization utilities for :mod:`weakincentives.session`."""

from __future__ import annotations

import json
import types
from collections.abc import Mapping
from dataclasses import dataclass, field, is_dataclass
from datetime import UTC, datetime
from importlib import import_module
from typing import Any

from ..serde import dump, parse

SNAPSHOT_SCHEMA_VERSION = "1"


class SnapshotSerializationError(RuntimeError):
    """Raised when snapshot capture fails due to unsupported payloads."""


class SnapshotRestoreError(RuntimeError):
    """Raised when snapshot restoration fails due to incompatible payloads."""


def normalize_snapshot_state(
    state: Mapping[type[Any], tuple[Any, ...]],
) -> Mapping[type[Any], tuple[Any, ...]]:
    """Validate snapshot state and return an immutable copy."""

    normalized: dict[type[Any], tuple[Any, ...]] = {}
    for slice_type, values in state.items():
        if not isinstance(slice_type, type):
            raise ValueError("Slice keys must be types")

        if not is_dataclass(slice_type):
            raise ValueError(f"Slice type {slice_type!r} is not a dataclass")

        items: list[Any] = []
        for value in values:
            if not is_dataclass(value) or isinstance(value, type):
                raise ValueError(
                    f"Slice {slice_type.__qualname__} contains non-dataclass value"
                )
            try:
                dump(value)
            except Exception as error:  # noqa: BLE001
                raise ValueError(
                    f"Slice {slice_type.__qualname__} cannot be serialized"
                ) from error
            items.append(value)

        normalized[slice_type] = tuple(items)

    return types.MappingProxyType(normalized)


def _type_identifier(cls: type[Any]) -> str:
    return f"{cls.__module__}:{cls.__qualname__}"


def _resolve_type(identifier: str) -> type[Any]:
    module_name, _, qualname = identifier.partition(":")
    if not module_name or not qualname:
        msg = f"Invalid type identifier: {identifier!r}"
        raise SnapshotRestoreError(msg)
    module = import_module(module_name)
    target: Any = module
    for part in qualname.split("."):
        target = getattr(target, part, None)
        if target is None:
            msg = f"Type {identifier!r} could not be resolved"
            raise SnapshotRestoreError(msg)
    if not isinstance(target, type):
        msg = f"Resolved object for {identifier!r} is not a type"
        raise SnapshotRestoreError(msg)
    return target


def _ensure_timezone(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt


def _infer_item_type(slice_type: type[Any], values: tuple[Any, ...]) -> type[Any]:
    if values:
        first_type = type(values[0])
        for value in values:
            if type(value) is not first_type:  # noqa: E721 - intentional identity check
                msg = (
                    "Snapshot slices must contain a single dataclass type; "
                    f"found {type(value)!r}"
                )
                raise SnapshotSerializationError(msg)
        return first_type
    return slice_type


@dataclass(slots=True, frozen=True)
class Snapshot:
    """Frozen value object representing session slice state."""

    created_at: datetime
    slices: Mapping[type[Any], tuple[Any, ...]] = field(
        default_factory=lambda: types.MappingProxyType({})
    )

    def __post_init__(self) -> None:
        normalized = {
            slice_type: tuple(values) for slice_type, values in self.slices.items()
        }
        object.__setattr__(self, "created_at", _ensure_timezone(self.created_at))
        object.__setattr__(self, "slices", types.MappingProxyType(normalized))

    def __hash__(self) -> int:
        ordered = tuple(
            sorted(
                self.slices.items(),
                key=lambda item: _type_identifier(item[0]),
            )
        )
        return hash((self.created_at, ordered))

    def to_json(self) -> str:
        """Serialize the snapshot to a JSON string."""

        payload_slices = []
        for slice_type, values in sorted(
            self.slices.items(), key=lambda item: _type_identifier(item[0])
        ):
            item_type = _infer_item_type(slice_type, values)
            try:
                serialized_items = [dump(value) for value in values]
            except Exception as error:  # noqa: BLE001
                msg = f"Failed to serialize slice {slice_type.__qualname__}"
                raise SnapshotSerializationError(msg) from error

            payload_slices.append(
                {
                    "slice_type": _type_identifier(slice_type),
                    "item_type": _type_identifier(item_type),
                    "items": serialized_items,
                }
            )

        payload = {
            "version": SNAPSHOT_SCHEMA_VERSION,
            "created_at": self.created_at.isoformat(),
            "slices": payload_slices,
        }
        return json.dumps(payload, sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> Snapshot:
        """Deserialize a snapshot from its JSON representation."""

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as error:
            raise SnapshotRestoreError("Invalid snapshot JSON") from error

        if not isinstance(payload, dict):
            raise SnapshotRestoreError("Snapshot payload must be an object")

        version = payload.get("version")
        if version != SNAPSHOT_SCHEMA_VERSION:
            msg = (
                "Snapshot schema version mismatch: "
                f"expected {SNAPSHOT_SCHEMA_VERSION}, got {version!r}"
            )
            raise SnapshotRestoreError(msg)

        created_at_raw = payload.get("created_at")
        if not isinstance(created_at_raw, str):
            raise SnapshotRestoreError("Snapshot created_at must be a string")
        try:
            created_at = datetime.fromisoformat(created_at_raw)
        except ValueError as error:
            raise SnapshotRestoreError("Invalid created_at timestamp") from error

        slices_payload = payload.get("slices", [])
        if not isinstance(slices_payload, list):
            raise SnapshotRestoreError("Snapshot slices must be a list")

        restored: dict[type[Any], tuple[Any, ...]] = {}
        for entry in slices_payload:
            if not isinstance(entry, dict):
                raise SnapshotRestoreError("Slice entry must be an object")

            slice_identifier = entry.get("slice_type")
            item_identifier = entry.get("item_type")
            items_payload = entry.get("items", [])

            if not isinstance(slice_identifier, str) or not isinstance(
                item_identifier, str
            ):
                raise SnapshotRestoreError("Slice type identifiers must be strings")

            slice_type = _resolve_type(slice_identifier)
            item_type = _resolve_type(item_identifier)

            if not is_dataclass(slice_type) or not is_dataclass(item_type):
                raise SnapshotRestoreError("Snapshot types must be dataclasses")

            if not isinstance(items_payload, list):
                raise SnapshotRestoreError("Slice items must be a list")

            try:
                restored_items = tuple(parse(item_type, item) for item in items_payload)
            except Exception as error:  # noqa: BLE001
                raise SnapshotRestoreError(
                    f"Failed to restore slice {slice_type.__qualname__}"
                ) from error

            restored[slice_type] = restored_items

        return cls(created_at=_ensure_timezone(created_at), slices=restored)
