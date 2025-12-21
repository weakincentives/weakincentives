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

"""Response parsing helpers for provider adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar, cast

from ..prompt.prompt import RenderedPrompt
from ..prompt.structured_output import OutputParseError, parse_structured_output
from ..types import JSONValue
from .core import PROMPT_EVALUATION_PHASE_RESPONSE, PromptEvaluationError
from .utilities import (
    extract_parsed_content,
    message_text_content,
    parse_schema_constrained_payload,
)

if TYPE_CHECKING:
    from ._provider_protocols import ProviderMessage

OutputT = TypeVar("OutputT")


@dataclass(slots=True)
class ResponseParser[OutputT]:
    """Handles parsing of provider responses into structured output."""

    prompt_name: str
    rendered: RenderedPrompt[OutputT]
    require_structured_output_text: bool
    _should_parse_structured_output: bool = field(init=False)

    def __post_init__(self) -> None:
        self._should_parse_structured_output = (
            self.rendered.output_type is not None
            and self.rendered.container is not None
        )

    def parse(
        self, message: ProviderMessage, provider_payload: dict[str, Any] | None
    ) -> tuple[OutputT | None, str | None]:
        """Parse the provider message into output and text content."""

        final_text = message_text_content(message.content)
        output: OutputT | None = None
        text_value: str | None = final_text or None

        if self._should_parse_structured_output:
            parsed_payload = extract_parsed_content(message)
            if parsed_payload is not None:
                try:
                    output = cast(
                        OutputT,
                        parse_schema_constrained_payload(
                            cast(JSONValue, parsed_payload), self.rendered
                        ),
                    )
                except (TypeError, ValueError) as error:
                    raise PromptEvaluationError(
                        str(error),
                        prompt_name=self.prompt_name,
                        phase=PROMPT_EVALUATION_PHASE_RESPONSE,
                        provider_payload=provider_payload,
                    ) from error
            elif final_text or not self.require_structured_output_text:
                try:
                    output = parse_structured_output(final_text or "", self.rendered)
                except OutputParseError as error:
                    raise PromptEvaluationError(
                        error.message,
                        prompt_name=self.prompt_name,
                        phase=PROMPT_EVALUATION_PHASE_RESPONSE,
                        provider_payload=provider_payload,
                    ) from error
            else:
                raise PromptEvaluationError(
                    "Provider response did not include structured output.",
                    prompt_name=self.prompt_name,
                    phase=PROMPT_EVALUATION_PHASE_RESPONSE,
                    provider_payload=provider_payload,
                )
            text_value = None

        return output, text_value

    @property
    def should_parse_structured_output(self) -> bool:
        return self._should_parse_structured_output


__all__ = ["ResponseParser"]
