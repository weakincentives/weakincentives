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

"""Core adapter interfaces shared across provider integrations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from ..prompts._types import SupportsDataclass
from ..prompts.prompt import Prompt
from ..prompts.tool import ToolResult

OutputT = TypeVar("OutputT")


class ProviderAdapter(Protocol[OutputT]):
    """Protocol describing the synchronous adapter contract."""

    def evaluate(
        self,
        prompt: Prompt[OutputT],
        *params: SupportsDataclass,
        parse_output: bool = True,
    ) -> PromptResponse[OutputT]:
        """Evaluate the prompt and return a structured response."""

        ...


@dataclass(slots=True)
class ToolCallRecord[ParamsT, ResultT]:
    """Record describing a single tool invocation during evaluation."""

    name: str
    params: ParamsT
    result: ToolResult[ResultT]
    call_id: str | None = None


@dataclass(slots=True)
class PromptResponse[OutputT]:
    """Structured result emitted by an adapter evaluation."""

    prompt_name: str
    text: str | None
    output: OutputT | None
    tool_results: tuple[ToolCallRecord[Any, Any], ...]
    provider_payload: dict[str, Any] | None = None


class PromptEvaluationError(RuntimeError):
    """Raised when evaluation against a provider fails."""

    def __init__(
        self,
        message: str,
        *,
        prompt_name: str,
        stage: str,
        provider_payload: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.prompt_name = prompt_name
        self.stage = stage
        self.provider_payload = provider_payload


__all__ = [
    "ProviderAdapter",
    "PromptEvaluationError",
    "PromptResponse",
    "ToolCallRecord",
]
