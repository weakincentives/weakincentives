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

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal, cast

from ..serde.parse import parse as parse_dataclass
from ..types import JSONValue
from ._structured_output_config import StructuredOutputConfig
from ._types import SupportsDataclass
from .protocols import RenderedPromptProtocol

__all__ = [
    "ARRAY_WRAPPER_KEY",
    "OutputParseError",
    "PayloadParsingConfig",
    "StructuredOutputConfig",
    "parse_structured_output",
]

ARRAY_WRAPPER_KEY: Final[str] = "items"

_JSON_FENCE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"```json\s*\n(.*?)```", re.IGNORECASE | re.DOTALL
)


class OutputParseError(Exception):
    """Raised when structured output parsing fails."""

    def __init__(
        self,
        message: str,
        *,
        dataclass_type: type[SupportsDataclass] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.dataclass_type = dataclass_type


@dataclass(frozen=True, slots=True)
class PayloadParsingConfig:
    """Configuration for parsing structured payloads into dataclasses."""

    container: Literal["object", "array"]
    allow_extra_keys: bool
    object_error: str
    array_error: str
    array_item_error: str


@dataclass(frozen=True, slots=True)
class DataclassPayloadParser[PayloadT: SupportsDataclass]:
    """Parse JSON payloads into dataclasses using a parsing configuration."""

    dataclass_type: type[PayloadT]
    config: PayloadParsingConfig

    @property
    def extra_mode(self) -> Literal["ignore", "forbid"]:
        return "ignore" if self.config.allow_extra_keys else "forbid"

    def parse(self, payload: JSONValue) -> PayloadT | list[PayloadT]:
        if self.config.container not in {"object", "array"}:
            raise TypeError("Unknown output container declared.")

        if self.config.container == "object":
            return self._parse_object_payload(payload)

        return self._parse_array_payload(payload)

    def _parse_object_payload(self, payload: JSONValue) -> PayloadT:
        if not isinstance(payload, Mapping):
            raise TypeError(self.config.object_error)

        mapping_payload = cast(Mapping[str, JSONValue], payload)
        return parse_dataclass(
            self.dataclass_type,
            mapping_payload,
            extra=self.extra_mode,
        )

    def _parse_array_payload(self, payload: JSONValue) -> list[PayloadT]:
        sequence_payload = self._normalize_array_payload(payload)
        parsed_items: list[PayloadT] = []
        for index, item in enumerate(sequence_payload):
            if not isinstance(item, Mapping):
                raise TypeError(self.config.array_item_error.format(index=index))
            mapping_item = cast(Mapping[str, JSONValue], item)
            parsed_item = parse_dataclass(
                self.dataclass_type,
                mapping_item,
                extra=self.extra_mode,
            )
            parsed_items.append(parsed_item)
        return parsed_items

    def _normalize_array_payload(self, payload: JSONValue) -> Sequence[JSONValue]:
        if isinstance(payload, Mapping):
            mapping_payload = cast(Mapping[str, JSONValue], payload)
            if ARRAY_WRAPPER_KEY not in mapping_payload:
                raise TypeError(self.config.array_error)
            payload = mapping_payload[ARRAY_WRAPPER_KEY]

        if not isinstance(payload, Sequence) or isinstance(
            payload, (str, bytes, bytearray)
        ):
            raise TypeError(self.config.array_error)

        return cast(Sequence[JSONValue], payload)


def parse_structured_output[PayloadT](
    output_text: str, rendered: RenderedPromptProtocol[PayloadT]
) -> PayloadT:
    """Parse a model response into the structured output type declared by the prompt."""

    config = rendered.structured_output
    if config is None:
        raise OutputParseError("Prompt does not declare structured output.")

    dataclass_type = config.dataclass_type
    payload = _extract_json_payload(output_text, dataclass_type)
    parsing_config = PayloadParsingConfig(
        container=config.container,
        allow_extra_keys=config.allow_extra_keys,
        object_error="Expected top-level JSON object.",
        array_error="Expected top-level JSON array.",
        array_item_error="Array item at index {index} is not an object.",
    )
    parser = DataclassPayloadParser(
        dataclass_type=dataclass_type, config=parsing_config
    )
    try:
        parsed = parser.parse(payload)
    except (TypeError, ValueError) as error:
        raise OutputParseError(str(error), dataclass_type=dataclass_type) from error

    return cast(PayloadT, parsed)


def _extract_json_payload(
    text: str, dataclass_type: type[SupportsDataclass]
) -> JSONValue:
    fenced_match = _JSON_FENCE_PATTERN.search(text)
    if fenced_match is not None:
        block = fenced_match.group(1).strip()
        try:
            return json.loads(block)
        except json.JSONDecodeError as error:
            raise OutputParseError(
                "Failed to decode JSON from fenced code block.",
                dataclass_type=dataclass_type,
            ) from error

    stripped = text.strip()
    if stripped:
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character not in "{[":
            continue
        try:
            payload, _ = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            continue
        return cast(JSONValue, payload)

    raise OutputParseError(
        "No JSON object or array found in assistant message.",
        dataclass_type=dataclass_type,
    )
