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

#: Key used when wrapping array outputs in an object container.
#: When a prompt expects an array of dataclasses, some models may return
#: `{"items": [...]}` instead of a bare array. This key is used to unwrap
#: such payloads automatically during parsing.
ARRAY_WRAPPER_KEY: Final[str] = "items"

_JSON_FENCE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"```json\s*\n(.*?)```", re.IGNORECASE | re.DOTALL
)


class OutputParseError(WinkError):
    """Raised when structured output parsing fails.

    This exception is raised when the model's response cannot be parsed into
    the expected dataclass type. Common causes include:

    - No JSON found in the response text
    - Invalid JSON syntax in fenced code blocks
    - JSON structure doesn't match the expected container type (object vs array)
    - Field validation errors during dataclass deserialization

    Attributes:
        message: Human-readable description of the parsing failure.
        dataclass_type: The target dataclass type that parsing was attempting
            to produce, or None if parsing failed before the type was determined.

    Example:
        >>> try:
        ...     result = parse_structured_output(response, rendered_prompt)
        ... except OutputParseError as e:
        ...     print(f"Failed to parse {e.dataclass_type}: {e.message}")
    """

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
    """Configuration for parsing structured JSON payloads into dataclasses.

    This configuration controls how raw JSON payloads are validated and parsed
    into typed dataclass instances. It specifies the expected container type,
    whether extra keys are permitted, and custom error messages for various
    failure modes.

    Attributes:
        container: Expected top-level JSON structure. Use ``"object"`` when
            expecting a single dataclass instance, or ``"array"`` when expecting
            a list of dataclass instances.
        allow_extra_keys: If True, extra keys in the JSON that don't map to
            dataclass fields are silently ignored. If False, extra keys raise
            a validation error.
        object_error: Error message to display when ``container="object"`` but
            the payload is not a JSON object.
        array_error: Error message to display when ``container="array"`` but
            the payload is not a JSON array (or wrapped array).
        array_item_error: Error message template for invalid array items. Use
            ``{index}`` placeholder for the item's position (e.g.,
            ``"Item at index {index} is not an object."``).

    Example:
        >>> config = PayloadParsingConfig(
        ...     container="array",
        ...     allow_extra_keys=False,
        ...     object_error="Expected a JSON object.",
        ...     array_error="Expected a JSON array.",
        ...     array_item_error="Array item {index} is invalid.",
        ... )
    """

    container: Literal["object", "array"]
    allow_extra_keys: bool
    object_error: str
    array_error: str
    array_item_error: str


def parse_structured_output[PayloadT](
    output_text: str, rendered: RenderedPromptProtocol[PayloadT]
) -> PayloadT:
    """Parse a model response into the structured output type declared by the prompt.

    Extracts JSON from the model's response text and deserializes it into the
    dataclass type specified by the rendered prompt's structured output config.
    Supports multiple JSON formats:

    1. Fenced code blocks: ````` ```json {...} ``` `````
    2. Raw JSON as the entire response
    3. Embedded JSON objects/arrays within prose text

    For array containers, automatically unwraps payloads wrapped in an object
    with the key specified by :data:`ARRAY_WRAPPER_KEY` (``"items"``).

    Args:
        output_text: The raw text response from the model, potentially containing
            JSON within prose, code fences, or as the complete response.
        rendered: A rendered prompt that declares a structured output type via
            its ``structured_output`` configuration. The prompt's output type
            determines the target dataclass for deserialization.

    Returns:
        The parsed dataclass instance (for object containers) or list of
        instances (for array containers), typed according to the prompt's
        declared output type.

    Raises:
        OutputParseError: If the prompt has no structured output config, no valid
            JSON is found in the response, the JSON structure doesn't match the
            expected container type, or dataclass validation fails.

    Example:
        >>> rendered = my_prompt.bind(resources).render()
        >>> response_text = '```json\\n{"name": "test", "value": 42}\\n```'
        >>> result = parse_structured_output(response_text, rendered)
        >>> print(result.name)
        test
    """

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
    """Parse a JSON payload into a dataclass instance or list of instances.

    Lower-level parsing function that converts already-extracted JSON data into
    typed dataclass instances. Unlike :func:`parse_structured_output`, this
    function expects pre-parsed JSON (not raw text) and requires explicit
    configuration rather than deriving it from a rendered prompt.

    For object containers, parses a single JSON object into one dataclass
    instance. For array containers, parses each item in the array into a
    separate dataclass instance, returning a list.

    Array payloads may be provided directly as a JSON array or wrapped in an
    object with the :data:`ARRAY_WRAPPER_KEY` (``"items"``) key. Both formats
    are accepted and unwrapped automatically.

    Args:
        dataclass_type: The target dataclass type for deserialization. Must be
            a dataclass compatible with the serde parsing system.
        payload: Pre-parsed JSON value (dict, list, or primitive). For object
            containers, should be a mapping. For array containers, should be
            a sequence or a mapping with an ``"items"`` key.
        config: Parsing configuration specifying the expected container type,
            extra key handling, and error messages.

    Returns:
        For ``container="object"``: A single dataclass instance.
        For ``container="array"``: A list of dataclass instances.

    Raises:
        TypeError: If the payload structure doesn't match the expected container
            type, array items aren't objects, or the container type is unknown.
        ValueError: If dataclass field validation fails during parsing.

    Example:
        >>> from weakincentives.prompt.structured_output import (
        ...     PayloadParsingConfig, parse_dataclass_payload
        ... )
        >>> config = PayloadParsingConfig(
        ...     container="object",
        ...     allow_extra_keys=True,
        ...     object_error="Expected object",
        ...     array_error="Expected array",
        ...     array_item_error="Invalid item at {index}",
        ... )
        >>> payload = {"name": "example", "count": 5}
        >>> result = parse_dataclass_payload(MyDataclass, payload, config)
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
        return parse_dataclass(dataclass_type, mapping_payload, extra=extra_mode)

    sequence_payload = _unwrap_array_payload(payload, config)
    parsed_items: list[ParseableDataclassT] = []
    for index, item in enumerate(sequence_payload):
        if not isinstance(item, Mapping):
            raise TypeError(config.array_item_error.format(index=index))
        mapping_item = cast(Mapping[str, JSONValue], item)
        parsed_items.append(
            parse_dataclass(dataclass_type, mapping_item, extra=extra_mode)
        )
    return parsed_items
