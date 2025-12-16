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

"""Shared helpers for serialising tool results into provider messages."""

from __future__ import annotations

from typing import Final

from ..prompt.tool_result import ToolResult, render_tool_payload
from ..types import SupportsToolResult

_UNSET: Final = object()


def serialize_tool_message(
    result: ToolResult[SupportsToolResult], *, payload: object = _UNSET
) -> str:
    """Return the text body sent with provider tool messages."""

    if result.exclude_value_from_context:
        return result.message

    value = result.value if payload is _UNSET else payload
    rendered_output = render_tool_payload(value)
    if rendered_output:
        if result.message:
            return f"{result.message}\n\n{rendered_output}"
        return rendered_output
    return result.message


__all__ = ["serialize_tool_message"]
