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

"""Structural typing primitives shared across provider adapters."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from ..prompt._types import SupportsDataclass
from ..prompt.tool_result import ToolResult

__all__ = [
    "ProviderChoice",
    "ProviderCompletionCallable",
    "ProviderCompletionResponse",
    "ProviderFunctionCall",
    "ProviderMessage",
    "ProviderToolCall",
    "ToolArgumentsParser",
    "ToolMessageSerializer",
]


class ProviderFunctionCall(Protocol):
    """Structural Protocol describing a provider function call payload."""

    name: str
    arguments: str | None


class ProviderToolCall(Protocol):
    """Structural Protocol describing a provider tool call payload."""

    @property
    def function(self) -> ProviderFunctionCall: ...


class ProviderMessage(Protocol):
    """Structural Protocol describing a provider message payload."""

    content: str | Sequence[object] | None
    tool_calls: Sequence[ProviderToolCall] | None


class ProviderChoice(Protocol):
    """Structural Protocol describing a provider choice payload."""

    @property
    def message(self) -> ProviderMessage: ...


class ProviderCompletionResponse(Protocol):
    """Structural Protocol describing a provider completion response."""

    choices: Sequence[ProviderChoice]


class ProviderCompletionCallable(Protocol):
    """Structural Protocol describing a provider completion callable."""

    def __call__(
        self, *args: object, **kwargs: object
    ) -> ProviderCompletionResponse: ...


class ToolArgumentsParser(Protocol):
    """Callable protocol responsible for parsing provider tool arguments."""

    def __call__(
        self,
        arguments_json: str | None,
        *,
        prompt_name: str,
        provider_payload: dict[str, Any] | None,
    ) -> dict[str, Any]: ...


class ToolMessageSerializer(Protocol):
    """Callable protocol responsible for serializing tool messages."""

    def __call__(
        self,
        result: ToolResult[SupportsDataclass],
        *,
        payload: object | None = ...,
    ) -> object: ...
