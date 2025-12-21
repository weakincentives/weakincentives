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

"""Shared helpers for provider adapters.

This module re-exports symbols from specialized submodules for backward
compatibility. New code should import directly from the submodules:

- throttle: ThrottlePolicy, ThrottleError, ThrottleDetails, throttle_details
- utilities: tool_to_spec, extract_payload, token_usage_from_payload, etc.
- response_parser: ResponseParser, build_json_schema_response_format, etc.
- tool_executor: ToolExecutor, ToolExecutionOutcome, tool_execution, etc.
- inner_loop: InnerLoop, InnerLoopConfig, InnerLoopInputs, run_inner_loop
"""

from __future__ import annotations

import json
import random
import time

# Re-export from _names for backward compatibility
from ._names import LITELLM_ADAPTER_NAME, OPENAI_ADAPTER_NAME, AdapterName

# Re-export from _provider_protocols for backward compatibility
from ._provider_protocols import (
    ProviderChoice,
    ProviderCompletionCallable,
    ProviderCompletionResponse,
    ProviderFunctionCall,
    ProviderMessage,
    ProviderToolCall,
)

# Re-export from inner_loop
from .inner_loop import (
    ChoiceSelector,
    InnerLoop,
    InnerLoopConfig,
    InnerLoopInputs,
    ProviderCall,
    run_inner_loop,
)

# Re-export from response_parser
from .response_parser import (
    ResponseParser,
    build_json_schema_response_format,
    content_part_text,
    extract_parsed_content,
    message_text_content,
    parse_schema_constrained_payload,
    parsed_payload_from_part,
)

# Re-export from throttle
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

# Re-export from tool_executor
from .tool_executor import (
    RejectedToolParams,
    ToolExecutionContext,
    ToolExecutionOutcome,
    ToolExecutor,
    ToolMessageSerializer,
    execute_tool_call,
    parse_tool_params,
    publish_tool_invocation,
    tool_execution,
)

# Re-export from utilities
from .utilities import (
    AdapterRenderContext,
    AdapterRenderOptions,
    ToolArgumentsParser,
    ToolChoice,
    build_resources,
    call_provider_with_normalization,
    deadline_provider_payload,
    extract_payload,
    first_choice,
    format_publish_failures,
    mapping_to_str_dict,
    parse_tool_arguments,
    prepare_adapter_conversation,
    raise_tool_deadline_error,
    serialize_tool_call,
    token_usage_from_payload,
    tool_to_spec,
)

# Backward-compatible aliases for private function names used by tests
_build_resources = build_resources
_content_part_text = content_part_text
_details_from_error = details_from_error
_jittered_backoff = jittered_backoff
_mapping_to_str_dict = mapping_to_str_dict
_parse_tool_params = parse_tool_params
_parsed_payload_from_part = parsed_payload_from_part
_publish_tool_invocation = publish_tool_invocation
_raise_tool_deadline_error = raise_tool_deadline_error
_RejectedToolParams = RejectedToolParams
_sleep_for = sleep_for
_ToolExecutionContext = ToolExecutionContext


__all__ = (
    "AdapterName",
    "AdapterRenderContext",
    "AdapterRenderOptions",
    "build_json_schema_response_format",
    "_build_resources",
    "build_resources",
    "call_provider_with_normalization",
    "ChoiceSelector",
    "_content_part_text",
    "content_part_text",
    "deadline_provider_payload",
    "_details_from_error",
    "details_from_error",
    "execute_tool_call",
    "extract_parsed_content",
    "extract_payload",
    "first_choice",
    "format_publish_failures",
    "InnerLoop",
    "InnerLoopConfig",
    "InnerLoopInputs",
    "_jittered_backoff",
    "jittered_backoff",
    "json",
    "LITELLM_ADAPTER_NAME",
    "_mapping_to_str_dict",
    "mapping_to_str_dict",
    "message_text_content",
    "new_throttle_policy",
    "OPENAI_ADAPTER_NAME",
    "parse_schema_constrained_payload",
    "parse_tool_arguments",
    "_parse_tool_params",
    "parse_tool_params",
    "_parsed_payload_from_part",
    "parsed_payload_from_part",
    "prepare_adapter_conversation",
    "ProviderCall",
    "ProviderChoice",
    "ProviderCompletionCallable",
    "ProviderCompletionResponse",
    "ProviderFunctionCall",
    "ProviderMessage",
    "ProviderToolCall",
    "_publish_tool_invocation",
    "publish_tool_invocation",
    "_raise_tool_deadline_error",
    "raise_tool_deadline_error",
    "random",
    "RejectedToolParams",
    "_RejectedToolParams",
    "ResponseParser",
    "run_inner_loop",
    "serialize_tool_call",
    "_sleep_for",
    "sleep_for",
    "throttle_details",
    "ThrottleDetails",
    "ThrottleError",
    "ThrottleKind",
    "ThrottlePolicy",
    "time",
    "token_usage_from_payload",
    "tool_execution",
    "tool_to_spec",
    "ToolArgumentsParser",
    "ToolChoice",
    "ToolExecutionContext",
    "_ToolExecutionContext",
    "ToolExecutionOutcome",
    "ToolExecutor",
    "ToolMessageSerializer",
)
