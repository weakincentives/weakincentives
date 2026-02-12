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

"""JSON schema normalization for Claude Agent SDK structured output.

Claude's structured output API has specific requirements for JSON schemas.
These functions normalize serde-generated schemas into a compatible form,
primarily by collapsing ``anyOf`` nullable unions into ``type=[..., "null"]``.
"""

from __future__ import annotations

from typing import Any, cast


def _collapse_nullable_any_of(any_of: object) -> dict[str, Any] | None:
    """Collapse ``anyOf`` nullable unions into ``type=[..., "null"]``."""
    nullable_arity = 2
    if not isinstance(any_of, list) or len(any_of) != nullable_arity:
        return None

    if not all(isinstance(entry, dict) for entry in any_of):
        return None

    entries = cast(list[dict[str, Any]], any_of)
    null_index = next(
        (
            index
            for index, entry in enumerate(entries)
            if entry.get("type") == "null" and len(entry) == 1
        ),
        None,
    )
    if null_index is None:
        return None

    non_null_entry = dict(entries[1 - null_index])
    schema_type = non_null_entry.get("type")
    if isinstance(schema_type, str):
        non_null_entry["type"] = [schema_type, "null"]
        return non_null_entry
    if isinstance(schema_type, list) and all(
        isinstance(value, str) for value in schema_type
    ):
        typed_values = cast(list[str], schema_type)
        non_null_entry["type"] = (
            typed_values if "null" in typed_values else [*typed_values, "null"]
        )
        return non_null_entry
    return None


def _normalize_claude_output_schema(raw_schema: dict[str, Any]) -> dict[str, Any]:
    """Normalize serde JSON schema for Claude structured output compatibility."""
    normalized = dict(raw_schema)

    if normalized.get("type") == "object":
        properties = normalized.get("properties")
        if isinstance(properties, dict):
            normalized["properties"] = {
                key: _normalize_claude_output_schema(cast(dict[str, Any], value))
                if isinstance(value, dict)
                else value
                for key, value in properties.items()
            }

    if "items" in normalized and isinstance(normalized["items"], dict):
        normalized["items"] = _normalize_claude_output_schema(
            cast(dict[str, Any], normalized["items"])
        )

    for combinator in ("anyOf", "oneOf", "allOf"):
        combinator_items = normalized.get(combinator)
        if isinstance(combinator_items, list):
            normalized[combinator] = [
                _normalize_claude_output_schema(cast(dict[str, Any], entry))
                if isinstance(entry, dict)
                else entry
                for entry in combinator_items
            ]

    for defs_key in ("$defs", "definitions"):
        defs = normalized.get(defs_key)
        if isinstance(defs, dict):
            normalized[defs_key] = {
                key: _normalize_claude_output_schema(cast(dict[str, Any], value))
                if isinstance(value, dict)
                else value
                for key, value in defs.items()
            }

    collapsed_nullable = _collapse_nullable_any_of(normalized.get("anyOf"))
    return collapsed_nullable if collapsed_nullable is not None else normalized
