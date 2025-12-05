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

from collections.abc import Sequence

from ._types import ComponentKey

SectionPath = tuple[ComponentKey, ...]


def _normalize_section_path(section_path: Sequence[str] | None) -> SectionPath:
    if section_path is None:
        return ()
    return tuple(ComponentKey(part) for part in section_path)


class PromptError(Exception):
    """Base class for prompt-related failures providing structured context."""

    def __init__(
        self,
        message: str,
        *,
        section_path: Sequence[str] | None = None,
        dataclass_type: type | None = None,
        placeholder: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.section_path: SectionPath = _normalize_section_path(section_path)
        self.dataclass_type = dataclass_type
        self.placeholder = placeholder


class PromptValidationError(PromptError):
    """Raised when prompt construction validation fails."""


class PromptRenderError(PromptError):
    """Raised when rendering a prompt fails."""


__all__ = [
    "PromptError",
    "PromptRenderError",
    "PromptValidationError",
    "SectionPath",
]
