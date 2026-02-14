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
from typing import Final, Literal, cast

from ..dataclasses import FrozenDataclass
from ..errors import WinkError
from ..serde import SerdeScope
from ..serde.parse import parse as parse_dataclass
from ..types import JSONValue, ParseableDataclassT
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


class OutputParseError(WinkError):
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


@FrozenDataclass()
class PayloadParsingConfig:
    """Configuration for parsing structured payloads into dataclasses."""

    container: Literal["object", "array"]
    allow_extra_keys: bool
    object_error: str
    array_error: str
    array_item_error: str


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
    try:
        parsed = parse_dataclass_payload(
            dataclass_type,
            payload,
            parsing_config,
        )
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


def _unwrap_array_payload(
    payload: JSONValue, config: PayloadParsingConfig
) -> Sequence[JSONValue]:
    """Unwrap array payload from possible wrapper object."""
    if isinstance(payload, Mapping):
        mapping_payload = cast(Mapping[str, JSONValue], payload)
        if ARRAY_WRAPPER_KEY not in mapping_payload:
            raise TypeError(config.array_error)
        payload = mapping_payload[ARRAY_WRAPPER_KEY]
    if not isinstance(payload, Sequence) or isinstance(
        payload, (str, bytes, bytearray)
    ):
        raise TypeError(config.array_error)
    return cast(Sequence[JSONValue], payload)


def parse_dataclass_payload(
    dataclass_type: type[ParseableDataclassT],
    payload: JSONValue,
    config: PayloadParsingConfig,
) -> ParseableDataclassT | list[ParseableDataclassT]:
    """Parse a JSON payload into dataclass instance(s) for structured output.

    Uses STRUCTURED_OUTPUT scope to skip fields marked with HiddenInStructuredOutput.
    """
    if config.container not in {"object", "array"}:
        raise TypeError("Unknown output container declared.")

    extra_mode: Literal["ignore", "forbid"] = (
        "ignore" if config.allow_extra_keys else "forbid"
    )

    if config.container == "object":
        if not isinstance(payload, Mapping):
            raise TypeError(config.object_error)
        mapping_payload = cast(Mapping[str, JSONValue], payload)
        return parse_dataclass(
            dataclass_type,
            mapping_payload,
            extra=extra_mode,
            scope=SerdeScope.STRUCTURED_OUTPUT,
        )

    sequence_payload = _unwrap_array_payload(payload, config)
    parsed_items: list[ParseableDataclassT] = []
    for index, item in enumerate(sequence_payload):
        if not isinstance(item, Mapping):
            raise TypeError(config.array_item_error.format(index=index))
        mapping_item = cast(Mapping[str, JSONValue], item)
        parsed_items.append(
            parse_dataclass(
                dataclass_type,
                mapping_item,
                extra=extra_mode,
                scope=SerdeScope.STRUCTURED_OUTPUT,
            )
        )
    return parsed_items
