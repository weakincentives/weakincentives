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

"""Unit tests for override helpers used by built-in tool suites."""

from __future__ import annotations

from weakincentives.tools._overrides import resolve_tool_accepts_overrides


def test_resolve_respects_default_when_overrides_missing() -> None:
    assert resolve_tool_accepts_overrides("tool", None, default=False) is False


def test_resolve_applies_boolean_overrides() -> None:
    assert resolve_tool_accepts_overrides("tool", True, default=False) is True


def test_resolve_reads_mapping_entries() -> None:
    overrides = {"tool": True, "other": False}

    assert resolve_tool_accepts_overrides("tool", overrides, default=False) is True
    assert resolve_tool_accepts_overrides("missing", overrides, default=True) is True


def test_resolve_handles_string_name() -> None:
    assert resolve_tool_accepts_overrides("tool", "tool", default=False) is True
    assert resolve_tool_accepts_overrides("tool", "other", default=True) is True


def test_resolve_checks_iterable_membership() -> None:
    assert (
        resolve_tool_accepts_overrides(
            "tool",
            {"tool"},
            default=False,
        )
        is True
    )
    assert (
        resolve_tool_accepts_overrides(
            "missing",
            {"tool"},
            default=True,
        )
        is True
    )
