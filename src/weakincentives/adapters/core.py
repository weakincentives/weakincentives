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

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from ..prompt._types import SupportsDataclass
from ..prompt.prompt import Prompt

OutputT = TypeVar("OutputT")


class EventBusProtocol(Protocol):
    def subscribe(
        self, event_type: type[object], handler: Callable[[object], None]
    ) -> None: ...

    def publish(self, event: object) -> object: ...


class SessionProtocol(Protocol):
    def rollback(self, snapshot: object) -> None: ...


class ToolInvocationProtocol(Protocol):
    @property
    def prompt_name(self) -> str: ...

    @property
    def adapter(self) -> str: ...

    @property
    def name(self) -> str: ...

    @property
    def params(self) -> SupportsDataclass: ...

    @property
    def result(self) -> object: ...

    @property
    def call_id(self) -> str | None: ...


class ProviderAdapter(Protocol[OutputT]):
    """Protocol describing the synchronous adapter contract."""

    def evaluate(
        self,
        prompt: Prompt[OutputT],
        *params: SupportsDataclass,
        parse_output: bool = True,
        bus: EventBusProtocol,
        session: SessionProtocol | None = None,
    ) -> PromptResponse[OutputT]:
        """Evaluate the prompt and return a structured response."""

        ...


@dataclass(slots=True)
class PromptResponse[OutputT]:
    """Structured result emitted by an adapter evaluation."""

    prompt_name: str
    text: str | None
    output: OutputT | None
    tool_results: tuple[ToolInvocationProtocol, ...]
    provider_payload: dict[str, Any] | None = None


class PromptEvaluationError(RuntimeError):
    """Raised when evaluation against a provider fails."""

    def __init__(
        self,
        message: str,
        *,
        prompt_name: str,
        phase: str,
        provider_payload: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.prompt_name = prompt_name
        self.phase = phase
        self.provider_payload = provider_payload


__all__ = [
    "PromptEvaluationError",
    "PromptResponse",
    "ProviderAdapter",
]
