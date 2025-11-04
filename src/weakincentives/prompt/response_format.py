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

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Literal

from .markdown import MarkdownSection

__all__ = ["ResponseFormatParams", "ResponseFormatSection"]


@dataclass(slots=True)
class ResponseFormatParams:
    """Parameter payload for the auto-generated response format section."""

    article: Literal["a", "an"]
    container: Literal["object", "array"]
    extra_clause: str


_RESPONSE_FORMAT_BODY: Final[
    str
] = """Return ONLY a single fenced JSON code block. Do not include any text
before or after the block.

The top-level JSON value MUST be ${article} ${container} that matches the fields
of the expected schema${extra_clause}"""


class ResponseFormatSection(MarkdownSection[ResponseFormatParams]):
    """Internal section that appends JSON-only response instructions."""

    def __init__(
        self,
        *,
        params: ResponseFormatParams,
        enabled: Callable[[ResponseFormatParams], bool] | None = None,
        accepts_overrides: bool = False,
    ) -> None:
        super().__init__(
            title="Response Format",
            key="response-format",
            template=_RESPONSE_FORMAT_BODY,
            default_params=params,
            enabled=enabled,
            accepts_overrides=accepts_overrides,
        )
