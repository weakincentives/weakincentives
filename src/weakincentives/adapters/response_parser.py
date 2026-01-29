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

"""Response parsing and structured output handling for provider adapters."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from ..prompt.prompt import RenderedPrompt
from ..prompt.structured_output import (
    ARRAY_WRAPPER_KEY,
    OutputParseError,
    PayloadParsingConfig,
    parse_dataclass_payload,
    parse_structured_output,
)
from ..serde import schema
from ..serde._scope import SerdeScope
from ..types import JSONValue
from ._provider_protocols import ProviderMessage
from .core import PROMPT_EVALUATION_PHASE_RESPONSE, PromptEvaluationError


def build_json_schema_response_format(
    rendered: RenderedPrompt[Any], prompt_name: str
) -> dict[str, JSONValue] | None:
    """Construct a JSON schema response format for structured outputs.

    Uses STRUCTURED_OUTPUT scope to exclude fields marked with HiddenInStructuredOutput.
    """

    output_type = rendered.output_type
    container = rendered.container
    allow_extra_keys = bool(rendered.allow_extra_keys)

    if output_type is None or container is None:
        return None

    extra_mode: Literal["ignore", "forbid"] = "ignore" if allow_extra_keys else "forbid"
    base_schema = schema(
        output_type, extra=extra_mode, scope=SerdeScope.STRUCTURED_OUTPUT
    )
    _ = base_schema.pop("title", None)

    if container == "array":
        schema_payload: dict[str, JSONValue] = {
            "type": "object",
            "properties": {
                ARRAY_WRAPPER_KEY: {
                    "type": "array",
                    "items": base_schema,
                }
            },
            "required": [ARRAY_WRAPPER_KEY],
        }
        if not allow_extra_keys:
            schema_payload["additionalProperties"] = False
    else:
        schema_payload = base_schema

    name = schema_name(prompt_name)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "schema": schema_payload,
        },
    }


def parse_schema_constrained_payload(
    payload: JSONValue, rendered: RenderedPrompt[Any]
) -> object:
    """Parse structured provider payloads constrained by prompt schema."""

    dataclass_type = rendered.output_type
    container = rendered.container
    allow_extra_keys = rendered.allow_extra_keys

    if dataclass_type is None or container is None:
        raise TypeError("Prompt does not declare structured output.")

    config = PayloadParsingConfig(
        container=container,
        allow_extra_keys=bool(allow_extra_keys),
        object_error="Expected provider payload to be a JSON object.",
        array_error="Expected provider payload to be a JSON array.",
        array_item_error="Array item at index {index} is not an object.",
    )
    return parse_dataclass_payload(dataclass_type, payload, config)


def message_text_content(content: object) -> str:
    """Extract text content from provider message payloads."""

    if isinstance(content, str) or content is None:
        return content or ""
    if isinstance(content, Sequence) and not isinstance(
        content, (str, bytes, bytearray)
    ):
        sequence_content = cast(
            Sequence[object],
            content,
        )
        fragments = [content_part_text(part) for part in sequence_content]
        return "".join(fragments)
    return str(content)


def extract_parsed_content(message: ProviderMessage) -> object | None:
    """Extract structured payloads surfaced directly by the provider."""

    parsed = getattr(message, "parsed", None)
    if parsed is not None:
        return parsed

    content = message.content
    if isinstance(content, Sequence) and not isinstance(
        content, (str, bytes, bytearray)
    ):
        for part in content:
            payload = parsed_payload_from_part(part)
            if payload is not None:
                return payload
    return None


def schema_name(prompt_name: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_-]+", "_", prompt_name.strip())
    cleaned = sanitized.strip("_") or "prompt"
    return f"{cleaned}_schema"


def content_part_text(part: object) -> str:
    if part is None:
        return ""
    if isinstance(part, Mapping):
        mapping_part = cast(Mapping[str, object], part)
        part_type = mapping_part.get("type")
        if part_type in {"output_text", "text"}:
            text_value = mapping_part.get("text")
            if isinstance(text_value, str):
                return text_value
        return ""
    part_type = getattr(part, "type", None)
    if part_type in {"output_text", "text"}:
        text_value = getattr(part, "text", None)
        if isinstance(text_value, str):
            return text_value
    return ""


def parsed_payload_from_part(part: object) -> object | None:
    if isinstance(part, Mapping):
        mapping_part = cast(Mapping[str, object], part)
        if mapping_part.get("type") == "output_json":
            return mapping_part.get("json")
        return None
    part_type = getattr(part, "type", None)
    if part_type == "output_json":
        return getattr(part, "json", None)
    return None


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
            # parse_structured_output and parse_schema_constrained_payload
            # always return a value or raise - output is never None here
            text_value = None

        return output, text_value

    @property
    def should_parse_structured_output(self) -> bool:
        return self._should_parse_structured_output


__all__ = [
    "ResponseParser",
    "build_json_schema_response_format",
    "extract_parsed_content",
    "message_text_content",
    "parse_schema_constrained_payload",
]
