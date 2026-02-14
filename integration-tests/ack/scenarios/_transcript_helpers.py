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

"""Shared transcript assertions for ACK scenarios."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from itertools import pairwise
from typing import cast

import pytest

from weakincentives.runtime.transcript import CANONICAL_ENTRY_TYPES


def collect_transcript_entries(
    caplog: pytest.LogCaptureFixture,
) -> list[dict[str, object]]:
    """Extract transcript entry contexts from caplog records."""
    entries: list[dict[str, object]] = []
    for record in caplog.records:
        event = getattr(record, "event", "")
        if event != "transcript.entry":
            continue
        context = getattr(record, "context", None)
        if isinstance(context, dict):
            entries.append(cast("dict[str, object]", dict(context)))
    return entries


def assert_envelope_complete(entry: Mapping[str, object]) -> None:
    """Assert required transcript envelope fields are present and valid."""
    prompt_name = entry.get("prompt_name")
    adapter = entry.get("adapter")
    entry_type = entry.get("entry_type")
    sequence_number = entry.get("sequence_number")
    source = entry.get("source")
    timestamp = entry.get("timestamp")

    assert isinstance(prompt_name, str) and prompt_name.strip()
    assert isinstance(adapter, str) and adapter.strip()
    assert isinstance(entry_type, str) and entry_type in CANONICAL_ENTRY_TYPES
    assert isinstance(sequence_number, int) and sequence_number >= 0
    assert isinstance(source, str) and source.strip()
    assert isinstance(timestamp, str) and timestamp.strip()

    _ = datetime.fromisoformat(timestamp)


def assert_sequence_monotonic(entries: list[dict[str, object]], source: str) -> None:
    """Assert strict + gapless sequence numbers for one transcript source."""
    source_sequences = [
        cast(int, entry["sequence_number"])
        for entry in entries
        if entry.get("source") == source
    ]
    assert source_sequences, f"No transcript entries for source {source}"
    for previous, current in pairwise(source_sequences):
        assert current == previous + 1, (
            f"Sequence for source {source} is not contiguous: {source_sequences}"
        )


def assert_entry_order(entries: list[dict[str, object]], *expected_types: str) -> None:
    """Assert expected entry types appear in order (subsequence match)."""
    actual_types = [cast(str, entry.get("entry_type", "")) for entry in entries]
    cursor = 0
    for expected in expected_types:
        while cursor < len(actual_types) and actual_types[cursor] != expected:
            cursor += 1
        assert cursor < len(actual_types), (
            f"Expected transcript type order {expected_types}, got {actual_types}"
        )
        cursor += 1


def assert_tool_use_before_result(
    entries: list[dict[str, object]],
    tool_name: str,
) -> None:
    """Assert tool_use for ``tool_name`` appears before matching tool_result.

    Matches tool names by suffix to handle bridge/MCP namespace prefixes
    (e.g. ``wink-tools_uppercase_text`` or ``mcp__wink__uppercase_text``
    both match ``uppercase_text``).
    """
    tool_use_index: int | None = None
    tool_result_index: int | None = None

    for index, entry in enumerate(entries):
        entry_type = entry.get("entry_type")
        extracted_name = _extract_tool_name(entry)

        if (
            entry_type == "tool_use"
            and _tool_name_matches(extracted_name, tool_name)
            and tool_use_index is None
        ):
            tool_use_index = index

        if entry_type == "tool_result" and (
            _tool_name_matches(extracted_name, tool_name) or extracted_name is None
        ):
            tool_result_index = index
            break

    assert tool_use_index is not None, f"No tool_use entry found for {tool_name}"
    assert tool_result_index is not None, f"No tool_result entry found for {tool_name}"
    assert tool_use_index < tool_result_index, (
        f"tool_use must appear before tool_result for {tool_name}: "
        f"tool_use={tool_use_index}, tool_result={tool_result_index}"
    )


def _tool_name_matches(extracted: str | None, expected: str) -> bool:
    """Check if extracted tool name matches the expected base name.

    Handles bridge/MCP namespace prefixes by checking suffix match.
    """
    if extracted is None:
        return False
    return extracted == expected or extracted.endswith(f"_{expected}")


def _extract_tool_name(entry: Mapping[str, object]) -> str | None:
    detail = entry.get("detail")
    if not isinstance(detail, Mapping):
        return None

    for key in ("tool_name", "name", "tool"):
        value = detail.get(key)
        if isinstance(value, str) and value:
            return value

    sdk_entry = detail.get("sdk_entry")
    if isinstance(sdk_entry, Mapping):
        for key in ("name", "tool_name"):
            value = sdk_entry.get(key)
            if isinstance(value, str) and value:
                return value

    notification = detail.get("notification")
    if isinstance(notification, Mapping):
        for key in ("tool", "tool_name", "name"):
            value = notification.get(key)
            if isinstance(value, str) and value:
                return value

    return None
