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
from collections.abc import Mapping
from typing import Any, Final, Literal, cast

from ..serde.dataclass_serde import parse as parse_dataclass
from .prompt import RenderedPrompt

__all__ = ["ARRAY_RESULT_KEY", "OutputParseError", "parse_output"]

ARRAY_RESULT_KEY: Final[str] = "items"

_JSON_FENCE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"```json\s*\n(.*?)```", re.IGNORECASE | re.DOTALL
)


class OutputParseError(Exception):
    """Raised when structured output parsing fails."""

    def __init__(
        self,
        message: str,
        *,
        dataclass_type: type[Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.dataclass_type = dataclass_type


def parse_output[PayloadT](
    output_text: str, rendered: RenderedPrompt[PayloadT]
) -> PayloadT:
    """Parse a model response into the structured output type declared by the prompt."""

    dataclass_type = rendered.output_type
    container = rendered.output_container
    allow_extra_keys = rendered.allow_extra_keys

    if dataclass_type is None or container is None:
        raise OutputParseError("Prompt does not declare structured output.")

    payload = _extract_json_payload(output_text, dataclass_type)
    extra_mode: Literal["ignore", "forbid"] = "ignore" if allow_extra_keys else "forbid"

    if container == "object":
        if not isinstance(payload, Mapping):
            raise OutputParseError(
                "Expected top-level JSON object.",
                dataclass_type=dataclass_type,
            )
        try:
            mapping_payload = cast(Mapping[str, object], payload)
            parsed = parse_dataclass(
                dataclass_type,
                mapping_payload,
                extra=extra_mode,
            )
        except (TypeError, ValueError) as error:
            raise OutputParseError(str(error), dataclass_type=dataclass_type) from error
        return cast(PayloadT, parsed)

    if container == "array":
        if isinstance(payload, Mapping):
            if ARRAY_RESULT_KEY not in payload:
                raise OutputParseError(
                    "Expected top-level JSON array.",
                    dataclass_type=dataclass_type,
                )
            payload = cast(Mapping[str, object], payload)[ARRAY_RESULT_KEY]
        if not isinstance(payload, list):
            raise OutputParseError(
                "Expected top-level JSON array.",
                dataclass_type=dataclass_type,
            )
        payload_list = cast(list[object], payload)
        parsed_items: list[Any] = []
        for index, item in enumerate(payload_list):
            if not isinstance(item, Mapping):
                raise OutputParseError(
                    f"Array item at index {index} is not an object.",
                    dataclass_type=dataclass_type,
                )
            try:
                mapping_item = cast(Mapping[str, object], item)
                parsed_item = parse_dataclass(
                    dataclass_type,
                    mapping_item,
                    extra=extra_mode,
                )
            except (TypeError, ValueError) as error:
                raise OutputParseError(
                    str(error), dataclass_type=dataclass_type
                ) from error
            parsed_items.append(parsed_item)
        return cast(PayloadT, parsed_items)

    raise OutputParseError(  # pragma: no cover - defensive guard
        "Unknown output container declared.",
        dataclass_type=dataclass_type,
    )


def _extract_json_payload(text: str, dataclass_type: type[Any]) -> object:
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
        except json.JSONDecodeError:  # pragma: no cover - defensive fallback
            continue
        return payload

    raise OutputParseError(
        "No JSON object or array found in assistant message.",
        dataclass_type=dataclass_type,
    )
