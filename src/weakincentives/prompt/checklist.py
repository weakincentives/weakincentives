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

"""Checklist section for domain-specific review checklists with progressive disclosure."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Self, cast, override

from ._types import SupportsDataclass
from ._visibility import SectionVisibility, VisibilitySelector
from .section import Section


@dataclass(slots=True, frozen=True)
class ChecklistItem:
    """A single item in a checklist.

    Attributes:
        text: The checklist item text to display.
        category: Optional category grouping for the item.
        severity: Importance level (critical, high, medium, low).
    """

    text: str
    category: str | None = None
    severity: str = "medium"


@dataclass(slots=True, frozen=True)
class ChecklistParams:
    """Parameters for checklist section rendering.

    Attributes:
        domain: The domain name for the checklist (e.g., "security", "performance").
        item_count: Number of items in the checklist, auto-populated during rendering.
    """

    domain: str = field(
        default="",
        metadata={"description": "Domain name for the checklist."},
    )
    item_count: int = field(
        default=0,
        metadata={"description": "Number of items in the checklist."},
    )


class ChecklistSection(Section[ChecklistParams]):
    """Section that renders domain-specific review checklists.

    This section extends the base Section to provide specialized rendering
    for checklist-style content with progressive disclosure support. In
    SUMMARY mode, it shows a brief overview with item count. In FULL mode,
    it renders all checklist items organized by category.

    Example:
        >>> items = [
        ...     ChecklistItem("Check for SQL injection", category="Input Validation"),
        ...     ChecklistItem("Validate user permissions", category="Authorization"),
        ... ]
        >>> section = ChecklistSection(
        ...     title="Security Checklist",
        ...     key="security-checklist",
        ...     domain="security",
        ...     items=items,
        ...     preamble="Review these security considerations:",
        ...     visibility=SectionVisibility.SUMMARY,
        ... )
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        title: str,
        key: str,
        domain: str,
        items: Sequence[ChecklistItem],
        preamble: str = "",
        children: Sequence[Section[SupportsDataclass]] | None = None,
        enabled: Callable[[SupportsDataclass], bool] | None = None,
        tools: Sequence[object] | None = None,
        accepts_overrides: bool = True,
        visibility: VisibilitySelector = SectionVisibility.FULL,
    ) -> None:
        """Initialize a checklist section.

        Args:
            title: Display title for the section.
            key: Unique identifier for the section.
            domain: Domain name (e.g., "security", "performance").
            items: Sequence of checklist items to render.
            preamble: Optional introductory text before the checklist.
            children: Optional nested sections.
            enabled: Optional predicate to conditionally enable the section.
            tools: Optional tools exposed by this section.
            accepts_overrides: Whether the section accepts prompt overrides.
            visibility: Default visibility (FULL or SUMMARY).
        """
        self._domain = domain
        self._items = tuple(items)
        self._preamble = preamble.strip()

        # Build the summary text
        item_count = len(self._items)
        categories = {item.category for item in self._items if item.category}
        category_hint = (
            f" ({len(categories)} categories)" if len(categories) > 1 else ""
        )
        summary = (
            f"{item_count} {domain} review items available{category_hint}. "
            "Request expansion for detailed checklist."
        )

        default_params = ChecklistParams(domain=domain, item_count=item_count)

        super().__init__(
            title=title,
            key=key,
            default_params=default_params,
            children=children,
            enabled=enabled,
            tools=tools,
            accepts_overrides=accepts_overrides,
            summary=summary,
            visibility=visibility,
        )

    @property
    def domain(self) -> str:
        """Return the checklist domain."""
        return self._domain

    @property
    def items(self) -> tuple[ChecklistItem, ...]:
        """Return the checklist items."""
        return self._items

    @property
    def preamble(self) -> str:
        """Return the preamble text."""
        return self._preamble

    @override
    def render(
        self,
        params: SupportsDataclass | None,
        depth: int,
        number: str,
        *,
        path: tuple[str, ...] = (),
        visibility: SectionVisibility | None = None,
    ) -> str:
        """Render the checklist section.

        In SUMMARY mode, renders a brief overview with item count.
        In FULL mode, renders all items organized by category.
        """
        effective = self.effective_visibility(override=visibility, params=params)
        heading = self._render_heading(depth, number, path)

        if effective == SectionVisibility.SUMMARY and self.summary is not None:
            return f"{heading}\n\n{self.summary}"

        return f"{heading}\n\n{self._render_full_body()}"

    def _render_heading(self, depth: int, number: str, path: tuple[str, ...]) -> str:
        """Render the section heading."""
        heading_level = "#" * (depth + 2)
        normalized_number = number.rstrip(".")
        path_str = ".".join(path) if path else ""
        title_with_path = (
            f"{self.title.strip()} ({path_str})" if path_str else self.title.strip()
        )
        return f"{heading_level} {normalized_number}. {title_with_path}"

    def _render_full_body(self) -> str:
        """Render the full checklist body with items grouped by category."""
        lines: list[str] = []

        if self._preamble:
            lines.append(self._preamble)
            lines.append("")

        # Group items by category
        categorized: dict[str | None, list[ChecklistItem]] = {}
        for item in self._items:
            category = item.category
            if category not in categorized:
                categorized[category] = []
            categorized[category].append(item)

        # Render items by category
        for category, category_items in categorized.items():
            if category is not None:
                lines.append(f"**{category}**")
                lines.append("")

            for item in category_items:
                severity_marker = self._severity_marker(item.severity)
                lines.append(f"- [ ] {severity_marker}{item.text}")

            lines.append("")

        return "\n".join(lines).strip()

    @staticmethod
    def _severity_marker(severity: str) -> str:
        """Return a marker prefix based on severity level."""
        markers = {
            "critical": "[CRITICAL] ",
            "high": "[HIGH] ",
            "medium": "",
            "low": "[LOW] ",
        }
        return markers.get(severity.lower(), "")

    @override
    def original_body_template(self) -> str | None:
        """Return a template representation for hashing."""
        # Build a stable template from items
        item_texts = [f"- {item.text}" for item in self._items]
        return "\n".join([self._preamble, *item_texts])

    @override
    def clone(self, **kwargs: object) -> Self:
        """Return a deep copy of the section."""
        cloned_children: list[Section[SupportsDataclass]] = []
        for child in self.children:
            if not hasattr(child, "clone"):
                raise TypeError(
                    "Section children must implement clone()."
                )  # pragma: no cover
            cloned_children.append(child.clone(**kwargs))

        cls: type[Any] = type(self)
        clone = cls(
            title=self.title,
            key=self.key,
            domain=self._domain,
            items=self._items,
            preamble=self._preamble,
            children=cloned_children,
            enabled=self._enabled,
            tools=self.tools(),
            accepts_overrides=self.accepts_overrides,
            visibility=self.visibility,
        )
        return cast(Self, clone)


__all__ = [
    "ChecklistItem",
    "ChecklistParams",
    "ChecklistSection",
]
