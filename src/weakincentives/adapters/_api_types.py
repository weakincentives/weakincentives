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

"""TypedDict definitions for LLM API payloads.

This module provides type-safe dictionary structures for API request/response
payloads used across provider adapters. Using TypedDict instead of dict[str, Any]
enables static type checking and IDE autocompletion.

Message Types:
    MessageDict: Base message with role and content
    AssistantMessageDict: Assistant message with optional tool calls
    ToolMessageDict: Tool response message
    FunctionCallOutputDict: OpenAI Responses API function call output

Tool Types:
    FunctionCallDict: Function call payload within tool calls
    ToolCallDict: Provider tool call structure
    FunctionCallItemDict: OpenAI Responses API function call item

Specification Types:
    ParametersSchemaDict: JSON Schema for tool parameters
    FunctionSpecDict: Function specification within tool spec
    ToolSpecDict: Complete tool specification

Payload Types:
    ProviderPayload: Generic provider response payload
    ToolArguments: Parsed tool call arguments
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Required, TypedDict

__all__ = [
    "AssistantMessageDict",
    "ClientKwargs",
    "FunctionCallDict",
    "FunctionCallItemDict",
    "FunctionCallOutputDict",
    "FunctionSpecDict",
    "LLMRequestParams",
    "MessageDict",
    "ParametersSchemaDict",
    "ProviderPayload",
    "ResponsesToolSpecDict",
    "ToolArguments",
    "ToolCallDict",
    "ToolMessageDict",
    "ToolSpecDict",
]


# -----------------------------------------------------------------------------
# Message Dictionaries
# -----------------------------------------------------------------------------


class MessageDict(TypedDict, total=False):
    """Base message dictionary for provider conversations.

    Used for system, user, assistant, and developer messages in the
    standard Chat Completions format.
    """

    role: Required[str]
    content: str | Sequence[object] | None
    name: str


class FunctionCallDict(TypedDict, total=False):
    """Function call payload within a tool call.

    Represents the function invocation details including name and
    JSON-encoded arguments string.
    """

    name: Required[str]
    arguments: str


class ToolCallDict(TypedDict, total=False):
    """Tool call structure in assistant messages.

    Represents a single tool invocation requested by the model.
    """

    id: str
    type: str
    function: Required[FunctionCallDict]


class AssistantMessageDict(TypedDict, total=False):
    """Assistant message with optional tool calls.

    Extends MessageDict with tool_calls field for assistant responses
    that include function invocations.
    """

    role: Required[str]
    content: str | Sequence[object] | None
    tool_calls: Sequence[ToolCallDict]


class ToolMessageDict(TypedDict, total=False):
    """Tool response message for Chat Completions format.

    Used to return tool execution results back to the model.
    The content field accepts the serialized tool result which
    may be a string or other JSON-serializable object depending
    on the provider.
    """

    role: Required[str]
    tool_call_id: Required[str]
    content: str | object


class FunctionCallOutputDict(TypedDict, total=False):
    """OpenAI Responses API function call output format.

    Used for returning tool results in the Responses API format
    which uses 'output' instead of 'content'.
    """

    type: Required[str]
    call_id: Required[str]
    output: str


class FunctionCallItemDict(TypedDict, total=False):
    """OpenAI Responses API function call item.

    Represents a function call in the Responses API input format.
    """

    type: Required[str]
    call_id: str
    name: Required[str]
    arguments: str


# -----------------------------------------------------------------------------
# Tool Specification Dictionaries
# -----------------------------------------------------------------------------


class ParametersSchemaDict(TypedDict, total=False):
    """JSON Schema for tool parameters.

    Defines the expected structure of tool input arguments using
    JSON Schema vocabulary.
    """

    type: Required[str]
    properties: dict[str, Any]
    required: Sequence[str]
    additionalProperties: bool


class FunctionSpecDict(TypedDict, total=False):
    """Function specification within a tool definition.

    Contains the function metadata used by the model to understand
    tool capabilities.
    """

    name: Required[str]
    description: str
    parameters: ParametersSchemaDict | Mapping[str, Any]
    strict: bool


class ToolSpecDict(TypedDict, total=False):
    """Complete tool specification for provider APIs.

    Standard format for declaring tools/functions available to the model.
    """

    type: Required[str]
    function: Required[FunctionSpecDict]


class ResponsesToolSpecDict(TypedDict, total=False):
    """OpenAI Responses API tool specification.

    Flattened format used by the Responses API where function fields
    are at the top level.
    """

    type: Required[str]
    name: Required[str]
    description: str
    parameters: ParametersSchemaDict | Mapping[str, Any]
    strict: bool


# -----------------------------------------------------------------------------
# Configuration Dictionaries
# -----------------------------------------------------------------------------


class LLMRequestParams(TypedDict, total=False):
    """LLM request parameters dictionary.

    Common parameters for LLM API requests that control generation
    behavior. All fields are optional.
    """

    temperature: float
    max_tokens: int
    max_output_tokens: int
    top_p: float
    presence_penalty: float
    frequency_penalty: float
    stop: list[str]
    seed: int
    n: int
    user: str
    logprobs: bool
    top_logprobs: int
    parallel_tool_calls: bool
    store: bool


class ClientKwargs(TypedDict, total=False):
    """Client constructor kwargs dictionary.

    Parameters passed to provider client constructors.
    """

    api_key: str
    base_url: str
    api_base: str
    organization: str
    timeout: float
    max_retries: int
    num_retries: int


# -----------------------------------------------------------------------------
# Payload Types
# -----------------------------------------------------------------------------


ProviderPayload = dict[str, Any]
"""Provider response payload dictionary.

Generic type for provider API response payloads that may contain
various fields depending on the provider and endpoint.
"""


ToolArguments = dict[str, Any]
"""Parsed tool call arguments dictionary.

Type for decoded JSON arguments from tool calls. Keys are parameter
names, values are the parsed JSON values.
"""
