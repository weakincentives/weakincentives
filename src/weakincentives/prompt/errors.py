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
from typing import TYPE_CHECKING, override

from ..errors import WinkError

if TYPE_CHECKING:
    from ._visibility import SectionVisibility

type SectionPath = tuple[str, ...]
"""Type alias for a section path within a prompt hierarchy.

A section path is a tuple of strings representing the hierarchical location
of a section. For example, ``("root", "child", "grandchild")`` identifies a
section nested three levels deep. An empty tuple ``()`` represents the root.
"""


def _normalize_section_path(section_path: Sequence[str] | None) -> SectionPath:
    if section_path is None:
        return ()
    return tuple(section_path)


class PromptError(WinkError):
    """Base class for prompt-related failures providing structured context.

    All prompt errors carry optional contextual information to help diagnose
    where and why the error occurred. Subclasses specialize for validation
    vs. rendering failures.

    Attributes:
        message: Human-readable description of the error.
        section_path: Hierarchical path to the section where the error occurred,
            as a tuple of strings. Empty tuple if not section-specific.
        dataclass_type: The dataclass type involved in the error, if applicable.
            Useful for debugging template parameter type mismatches.
        placeholder: The placeholder name that caused the error, if applicable.
            Helps identify which template variable failed.

    Example:
        >>> try:
        ...     prompt.render(params)
        ... except PromptError as e:
        ...     print(f"Error in section {'.'.join(e.section_path)}: {e.message}")
    """

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
    """Raised when prompt construction or configuration validation fails.

    This error indicates a problem with the prompt's structure or configuration
    that is detected before rendering. Common causes include:

    - Invalid section keys that don't match the required pattern
    - Duplicate section keys within the same parent
    - Missing required template parameters
    - Invalid parameter types for template placeholders
    - Circular section references

    This error typically indicates a programming error that should be fixed
    in the code, rather than a runtime condition to handle gracefully.

    Example:
        >>> from weakincentives.prompt import Section
        >>> # Invalid key raises PromptValidationError
        >>> Section(key="Invalid Key!")  # doctest: +SKIP
        PromptValidationError: Section key must match pattern...
    """


class PromptRenderError(PromptError):
    """Raised when rendering a prompt to its final output fails.

    This error occurs during the render phase when converting a prompt template
    and parameters into the final output. Common causes include:

    - Template placeholder that cannot be substituted (missing or wrong type)
    - Conditional section evaluation failure
    - Resource resolution failure during rendering
    - Output formatting or serialization errors

    Unlike ``PromptValidationError``, render errors may depend on runtime
    values and can sometimes be handled by adjusting parameters.

    Example:
        >>> try:
        ...     rendered = prompt.render(my_params)
        ... except PromptRenderError as e:
        ...     if e.placeholder:
        ...         print(f"Failed to render placeholder: {e.placeholder}")
    """


class VisibilityExpansionRequired(PromptError):
    """Raised when the model requests expansion of summarized sections.

    This exception signals that the model needs to see the full content of
    sections that were previously shown in summarized form. It is part of
    the progressive disclosure pattern where sections start collapsed and
    expand on demand.

    Callers should catch this exception, extract the visibility overrides,
    merge them with any existing overrides, and retry prompt evaluation.

    Attributes:
        requested_overrides: Mapping from section paths to their requested
            visibility states. Apply these overrides when re-rendering.
        reason: Human-readable explanation of why expansion was requested,
            typically provided by the model.
        section_keys: Tuple of section key strings that need expansion.
            Convenience accessor for the keys without full paths.

    Example:
        >>> overrides = {}
        >>> while True:
        ...     try:
        ...         result = evaluate_prompt(prompt, overrides)
        ...         break
        ...     except VisibilityExpansionRequired as e:
        ...         overrides.update(e.requested_overrides)
        ...         # Loop continues with expanded sections
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
    "VisibilityExpansionRequired",
]
