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

"""Message content extraction utilities for Claude Agent SDK adapter.

Standalone functions for extracting and formatting message content
from SDK response objects for debug logging and tracing.
"""

from __future__ import annotations

from typing import Any


def _extract_content_block(block: dict[str, Any]) -> dict[str, Any]:
    """Extract full content from a single content block for logging."""
    block_type = block.get("type")
    result: dict[str, Any] = {"type": block_type}

    if block_type == "tool_use":
        result["name"] = block.get("name", "unknown")
        result["id"] = block.get("id", "")
        # Include full input for complete tracing
        if "input" in block:
            result["input"] = block["input"]
    elif block_type == "text":
        result["text"] = block.get("text", "")
    elif block_type == "tool_result":
        result["tool_use_id"] = block.get("tool_use_id", "")
        result["content"] = block.get("content", "")
        if "is_error" in block:
            result["is_error"] = block["is_error"]
    else:
        # For other types, include the whole block
        result.update(block)

    return result


def _extract_list_content(content: list[Any]) -> list[dict[str, Any]]:
    """Extract full content from content block list."""
    return [
        _extract_content_block(block) for block in content if isinstance(block, dict)
    ]


def _extract_inner_message_content(inner_msg: dict[str, Any]) -> dict[str, Any]:
    """Extract full content from the inner message dict."""
    result: dict[str, Any] = {}
    role = inner_msg.get("role")
    if role:
        result["role"] = role
    content = inner_msg.get("content")
    if isinstance(content, str):
        result["content"] = content
    elif isinstance(content, list):
        result["content_blocks"] = _extract_list_content(content)
    return result


def _extract_message_content(message: Any) -> dict[str, Any]:  # noqa: ANN401
    """Extract full content from an SDK message for debug logging."""
    result: dict[str, Any] = {}

    # Try to get the inner message dict (common pattern in SDK messages)
    inner_msg = getattr(message, "message", None)
    if isinstance(inner_msg, dict):
        result.update(_extract_inner_message_content(inner_msg))

    # ResultMessage specific: extract the full result field
    sdk_result = getattr(message, "result", None)
    if sdk_result and isinstance(sdk_result, str):
        result["result"] = sdk_result

    # Structured output - include full content
    structured_output = getattr(message, "structured_output", None)
    if structured_output:
        result["structured_output"] = structured_output

    # Include usage if present
    usage = getattr(message, "usage", None)
    if usage:
        result["usage"] = usage if isinstance(usage, dict) else str(usage)
        # Extract thinking tokens for extended thinking mode
        if isinstance(usage, dict):
            result["input_tokens"] = usage.get("input_tokens")
            result["output_tokens"] = usage.get("output_tokens")
            # Check for cache-related fields
            result["cache_read_input_tokens"] = usage.get("cache_read_input_tokens")
            result["cache_creation_input_tokens"] = usage.get(
                "cache_creation_input_tokens"
            )

    # Extract thinking content if present (extended thinking mode)
    thinking = getattr(message, "thinking", None)
    if thinking:
        result["has_thinking"] = True
        if isinstance(thinking, str):  # pragma: no cover
            result["thinking_preview"] = thinking[:200]
            result["thinking_length"] = len(thinking)

    return result
