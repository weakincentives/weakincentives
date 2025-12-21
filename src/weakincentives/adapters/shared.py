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

"""Shared helpers for provider adapters."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import json as _json
import random as _random
import time as _time

from ._names import LITELLM_ADAPTER_NAME, OPENAI_ADAPTER_NAME, AdapterName
from ._provider_protocols import (
    ProviderChoice,
    ProviderCompletionCallable,
    ProviderCompletionResponse,
    ProviderFunctionCall,
    ProviderMessage,
    ProviderToolCall,
)
from .inner_loop import (
    ChoiceSelector,
    InnerLoop,
    InnerLoopConfig,
    InnerLoopInputs,
    ProviderCall,
    ToolMessageSerializer,
    run_inner_loop,
)
from .response_parser import ResponseParser
from .throttle import (
    ThrottleDetails,
    ThrottleError,
    ThrottleKind,
    ThrottlePolicy,
    details_from_error,
    jittered_backoff,
    new_throttle_policy,
    sleep_for,
    throttle_details,
)
from .tool_executor import (
    ToolExecutionOutcome,
    ToolExecutor,
    _parse_tool_params,
    _publish_tool_invocation,
    _RejectedToolParams,
    _ToolExecutionContext,
    execute_tool_call,
    tool_execution,
)
from .utilities import (
    AdapterRenderContext,
    AdapterRenderOptions,
    ToolArgumentsParser,
    ToolChoice,
    _content_part_text,
    _mapping_to_str_dict,
    _parsed_payload_from_part,
    build_json_schema_response_format,
    build_resources,
    call_provider_with_normalization,
    deadline_provider_payload,
    extract_parsed_content,
    extract_payload,
    first_choice,
    format_publish_failures,
    message_text_content,
    parse_schema_constrained_payload,
    parse_tool_arguments,
    prepare_adapter_conversation,
    raise_tool_deadline_error,
    serialize_tool_call,
    token_usage_from_payload,
    tool_to_spec,
)

# Compatibility aliases for private helpers used in tests and documentation.
_details_from_error = details_from_error
_jittered_backoff = jittered_backoff
_sleep_for = sleep_for
_build_resources = build_resources
_raise_tool_deadline_error = raise_tool_deadline_error
random = _random
time = _time
json = _json

__all__ = (  # noqa: RUF022
    "AdapterName",
    "AdapterRenderContext",
    "AdapterRenderOptions",
    "ChoiceSelector",
    "InnerLoop",
    "InnerLoopConfig",
    "InnerLoopInputs",
    "LITELLM_ADAPTER_NAME",
    "OPENAI_ADAPTER_NAME",
    "ProviderCall",
    "ProviderChoice",
    "ProviderCompletionCallable",
    "ProviderCompletionResponse",
    "ProviderFunctionCall",
    "ProviderMessage",
    "ProviderToolCall",
    "ResponseParser",
    "ThrottleDetails",
    "ThrottleError",
    "ThrottleKind",
    "ThrottlePolicy",
    "ToolArgumentsParser",
    "ToolChoice",
    "ToolExecutionOutcome",
    "ToolExecutor",
    "ToolMessageSerializer",
    "_RejectedToolParams",
    "_ToolExecutionContext",
    "_content_part_text",
    "_mapping_to_str_dict",
    "_parse_tool_params",
    "_parsed_payload_from_part",
    "_publish_tool_invocation",
    "build_json_schema_response_format",
    "call_provider_with_normalization",
    "deadline_provider_payload",
    "execute_tool_call",
    "extract_parsed_content",
    "extract_payload",
    "first_choice",
    "format_publish_failures",
    "message_text_content",
    "new_throttle_policy",
    "parse_schema_constrained_payload",
    "parse_tool_arguments",
    "prepare_adapter_conversation",
    "run_inner_loop",
    "serialize_tool_call",
    "throttle_details",
    "token_usage_from_payload",
    "tool_execution",
    "tool_to_spec",
)
