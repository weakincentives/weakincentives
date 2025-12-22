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

"""Shared utility functions for provider adapters.

This module re-exports utilities from focused submodules for convenient access:

- ``rendering``: Prompt rendering context and options
- ``token_usage``: Token metrics extraction
- ``tool_spec``: Tool specification building and serialization
- ``deadline_utils``: Deadline handling utilities
- ``provider_response``: Provider response processing
- ``resources``: Resource registry building
"""

from __future__ import annotations

from collections.abc import Sequence

from ..runtime.events import HandlerFailure
from .deadline_utils import deadline_provider_payload, raise_tool_deadline_error
from .provider_response import (
    call_provider_with_normalization,
    extract_payload,
    first_choice,
    mapping_to_str_dict,
)
from .rendering import (
    AdapterRenderContext,
    AdapterRenderOptions,
    prepare_adapter_conversation,
)
from .resources import build_resources
from .token_usage import token_usage_from_payload
from .tool_spec import (
    ToolArgumentsParser,
    ToolChoice,
    parse_tool_arguments,
    serialize_tool_call,
    tool_to_spec,
)


def format_dispatch_failures(failures: Sequence[HandlerFailure]) -> str:
    """Summarize dispatch failures encountered while applying tool results."""

    def _message_for(failure: HandlerFailure) -> str:
        message = str(failure.error).strip()
        return message or failure.error.__class__.__name__

    messages = [_message_for(failure) for failure in failures]

    if not messages:
        return "Reducer errors prevented applying tool result."

    joined = "; ".join(messages)
    return f"Reducer errors prevented applying tool result: {joined}"


__all__ = [
    "AdapterRenderContext",
    "AdapterRenderOptions",
    "ToolArgumentsParser",
    "ToolChoice",
    "build_resources",
    "call_provider_with_normalization",
    "deadline_provider_payload",
    "extract_payload",
    "first_choice",
    "format_dispatch_failures",
    "mapping_to_str_dict",
    "parse_tool_arguments",
    "prepare_adapter_conversation",
    "raise_tool_deadline_error",
    "serialize_tool_call",
    "token_usage_from_payload",
    "tool_to_spec",
]
