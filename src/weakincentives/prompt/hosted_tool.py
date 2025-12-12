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

"""Hosted tool abstraction for provider-executed capabilities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from .errors import PromptValidationError

_NAME_MIN_LENGTH: Final = 1
_NAME_MAX_LENGTH: Final = 64
_DESCRIPTION_MIN_LENGTH: Final = 1
_DESCRIPTION_MAX_LENGTH: Final = 200

_NAME_PATTERN: Final[re.Pattern[str]] = re.compile(
    rf"^[a-z0-9_-]{{{_NAME_MIN_LENGTH},{_NAME_MAX_LENGTH}}}$"
)


@dataclass(slots=True, frozen=True)
class HostedTool[ConfigT]:
    """A tool executed by the provider rather than locally.

    HostedTool is a minimal container for provider-specific configurations.
    ConfigT is a provider-specific configuration type defined in the adapter
    layer. Users import the appropriate config type from the adapter they
    are using.

    Attributes:
        kind: Codec lookup key (e.g., "web_search"). Identifies which codec
            handles serialization and output parsing for this tool.
        name: Registry key, unique within prompt. Must match the pattern
            `^[a-z0-9_-]{1,64}$`.
        description: Documentation for the tool (1-200 ASCII characters).
        config: Provider-specific configuration, opaque to the core library.
    """

    kind: str
    name: str
    description: str
    config: ConfigT

    def __post_init__(self) -> None:
        """Validate hosted tool fields."""
        self._validate_name()
        self._validate_description()

    def _validate_name(self) -> None:
        """Validate the tool name follows provider constraints."""
        raw_name = self.name
        stripped_name = raw_name.strip()
        if raw_name != stripped_name:
            raise PromptValidationError(
                "HostedTool name must not contain surrounding whitespace.",
                placeholder=stripped_name,
            )

        if not stripped_name:
            raise PromptValidationError(
                (
                    f"HostedTool name must match the function name constraints "
                    f"(1-{_NAME_MAX_LENGTH} lowercase ASCII letters, digits, "
                    f"underscores, or hyphens)."
                ),
                placeholder=stripped_name,
            )

        if len(stripped_name) > _NAME_MAX_LENGTH or not _NAME_PATTERN.fullmatch(
            stripped_name
        ):
            raise PromptValidationError(
                f"HostedTool name must match pattern: {_NAME_PATTERN.pattern}.",
                placeholder=stripped_name,
            )

    def _validate_description(self) -> None:
        """Validate the tool description."""
        description_clean = self.description.strip()
        if (
            not description_clean
            or len(description_clean) < _DESCRIPTION_MIN_LENGTH
            or len(description_clean) > _DESCRIPTION_MAX_LENGTH
        ):
            raise PromptValidationError(
                (
                    f"HostedTool description must be "
                    f"{_DESCRIPTION_MIN_LENGTH}-{_DESCRIPTION_MAX_LENGTH} ASCII characters."
                ),
                placeholder="description",
            )

        try:
            _ = description_clean.encode("ascii")
        except UnicodeEncodeError as error:
            raise PromptValidationError(
                "HostedTool description must be ASCII.",
                placeholder="description",
            ) from error


__all__ = ["HostedTool"]
