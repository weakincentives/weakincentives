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

from collections.abc import Mapping, Sequence
from typing import override

from ..errors import WinkError
from ._visibility import SectionVisibility

type SectionPath = tuple[str, ...]


def _normalize_section_path(section_path: Sequence[str] | None) -> SectionPath:
    if section_path is None:
        return ()
    return tuple(section_path)


class PromptError(WinkError):
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


class VisibilityExpansionRequired(PromptError):
    """Raised when the model requests expansion of summarized sections.

    Callers should catch this exception, extract the visibility overrides,
    and retry prompt evaluation with the requested sections expanded.
    """

    def __init__(
        self,
        message: str,
        *,
        requested_overrides: Mapping[SectionPath, SectionVisibility],
        reason: str,
        section_keys: tuple[str, ...],
    ) -> None:
        super().__init__(message)
        self.requested_overrides: Mapping[SectionPath, SectionVisibility] = (
            requested_overrides
        )
        self.reason = reason
        self.section_keys = section_keys

    @override
    def __str__(self) -> str:
        keys = ", ".join(".".join(p) for p in self.requested_overrides)
        return (
            f"Visibility expansion required for sections: {keys}. Reason: {self.reason}"
        )


__all__ = [
    "PromptError",
    "PromptRenderError",
    "PromptValidationError",
    "SectionPath",
    "SectionVisibility",
    "VisibilityExpansionRequired",
]
