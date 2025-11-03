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

"""Utilities for configuring override behavior on built-in tools."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import cast


def resolve_tool_accepts_overrides(
    name: str,
    overrides: bool | Iterable[str] | Mapping[str, bool] | None,
    *,
    default: bool,
) -> bool:
    """Return the override flag for a tool based on user configuration.

    The ``overrides`` parameter accepts multiple shapes so that callers can keep
    their ergonomics simple:

    - ``None``: fall back to ``default``.
    - ``bool``: apply the value to every tool in the section.
    - ``Mapping[str, bool]``: look up ``name`` and fall back to ``default`` when
      the tool is not present.
    - ``Iterable[str]``: enable overrides when ``name`` is present in the
      iterable. ``str`` values are treated as a single tool name.
    """

    if overrides is None:
        return default
    if isinstance(overrides, bool):
        return overrides
    if isinstance(overrides, Mapping):
        mapping = cast(Mapping[str, bool], overrides)
        value = mapping.get(name)
        return default if value is None else value
    if isinstance(overrides, str):
        return overrides == name or default
    return name in overrides or default


__all__ = ["resolve_tool_accepts_overrides"]
