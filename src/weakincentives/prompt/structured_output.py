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
from typing import Any, Final, Literal, Protocol, TypeVar, cast

from ..serde.parse import parse as parse_dataclass
from ._types import SupportsDataclass

__all__ = [
    "ARRAY_WRAPPER_KEY",
    "OutputParseError",
    "StructuredOutputSpec",
    "parse_structured_output",
]

ARRAY_WRAPPER_KEY: Final[str] = "items"

_JSON_FENCE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"```json\s*\n(.*?)```", re.IGNORECASE | re.DOTALL
)


DataclassT = TypeVar("DataclassT", bound=SupportsDataclass)


@dataclass(frozen=True, slots=True)
class StructuredOutputSpec[DataclassT]:
    """Resolved structured output declaration for a prompt."""

    dataclass_type: type[DataclassT]
    container: Literal["object", "array"]
    allow_extra_keys: bool


class StructuredRenderedPrompt[PayloadT](Protocol):
    @property
    def structured_output(self) -> StructuredOutputSpec[SupportsDataclass] | None:
        """Structured output metadata declared by the prompt."""


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


def parse_structured_output[PayloadT](
    output_text: str, rendered: StructuredRenderedPrompt[PayloadT]
) -> PayloadT:
    """Parse a model response into the structured output type declared by the prompt."""

    spec = rendered.structured_output
    if spec is None:
        raise OutputParseError("Prompt does not declare structured output.")

    dataclass_type = spec.dataclass_type
    container = spec.container
    allow_extra_keys = spec.allow_extra_keys
    payload = _extract_json_payload(output_text, dataclass_type)
    try:
        parsed = parse_dataclass_payload(
            dataclass_type,
            container,
            payload,
            allow_extra_keys=allow_extra_keys,
            object_error="Expected top-level JSON object.",
            array_error="Expected top-level JSON array.",
            array_item_error="Array item at index {index} is not an object.",
        )
    except (TypeError, ValueError) as error:
        raise OutputParseError(str(error), dataclass_type=dataclass_type) from error

    return cast(PayloadT, parsed)


def _extract_json_payload(text: str, dataclass_type: type[SupportsDataclass]) -> object:
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
        return payload

    raise OutputParseError(
        "No JSON object or array found in assistant message.",
        dataclass_type=dataclass_type,
    )


def parse_dataclass_payload(
    dataclass_type: type[SupportsDataclass],
    container: Literal["object", "array"],
    payload: object,
    *,
    allow_extra_keys: bool,
    object_error: str,
    array_error: str,
    array_item_error: str,
) -> object:
    if container not in {"object", "array"}:
        raise TypeError("Unknown output container declared.")

    extra_mode: Literal["ignore", "forbid"] = "ignore" if allow_extra_keys else "forbid"

    if container == "object":
        if not isinstance(payload, Mapping):
            raise TypeError(object_error)
        mapping_payload = cast(Mapping[str, object], payload)
        return parse_dataclass(dataclass_type, mapping_payload, extra=extra_mode)

    if isinstance(payload, Mapping):
        mapping_payload = cast(Mapping[str, object], payload)
        if ARRAY_WRAPPER_KEY not in mapping_payload:
            raise TypeError(array_error)
        payload = mapping_payload[ARRAY_WRAPPER_KEY]
    if not isinstance(payload, Sequence) or isinstance(
        payload, (str, bytes, bytearray)
    ):
        raise TypeError(array_error)
    sequence_payload = cast(Sequence[object], payload)
    parsed_items: list[Any] = []
    for index, item in enumerate(sequence_payload):
        if not isinstance(item, Mapping):
            raise TypeError(array_item_error.format(index=index))
        mapping_item = cast(Mapping[str, object], item)
        parsed_item = parse_dataclass(
            dataclass_type,
            mapping_item,
            extra=extra_mode,
        )
        parsed_items.append(parsed_item)
    return parsed_items
