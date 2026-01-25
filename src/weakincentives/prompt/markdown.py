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

import textwrap
from collections.abc import Callable, Sequence
from dataclasses import fields, is_dataclass
from string import Template
from typing import TYPE_CHECKING, Any, Self, TypeVar, cast, override

from ..serde import clone as clone_dataclass
from ..types.dataclass import (
    SupportsDataclass,
)
from .errors import PromptRenderError
from .section import Section, SectionVisibility, VisibilitySelector

if TYPE_CHECKING:
    from .policy import ToolPolicy

MarkdownParamsT = TypeVar("MarkdownParamsT", bound=SupportsDataclass, covariant=True)


class MarkdownSection(Section[MarkdownParamsT]):
    """A prompt section that renders markdown content using :class:`string.Template`.

    MarkdownSection is the primary building block for constructing prompts. It uses
    Python's ``string.Template`` syntax (``$placeholder`` or ``${placeholder}``) to
    substitute values from a parameters dataclass into the template text.

    Example::

        @dataclass(frozen=True)
        class AgentParams:
            name: str
            role: str

        section = MarkdownSection[AgentParams](
            title="Agent Identity",
            key="identity",
            template='''
            You are $name, a $role.
            Follow instructions carefully.
            ''',
            default_params=AgentParams(name="Assistant", role="helpful AI"),
        )

        # Render with default params
        output = section.render(params=None, depth=0, number="1")

        # Render with custom params
        custom = AgentParams(name="Claude", role="coding assistant")
        output = section.render(params=custom, depth=0, number="1")

    Sections can be nested via ``children`` to create hierarchical prompt structures,
    and can expose tools and policies that become available when the section is active.

    Attributes:
        template: The ``string.Template`` text with ``$placeholder`` substitutions.
        title: Display title shown in the rendered heading.
        key: Unique identifier for the section (used in paths and overrides).
        default_params: Fallback parameter values when none are provided.
        children: Nested child sections rendered after this section's body.
        accepts_overrides: Whether the template body can be overridden at render time.
        summary: Optional abbreviated template used when visibility is SUMMARY.
        visibility: Controls whether section renders as FULL, SUMMARY, or HIDDEN.
    """

    def __init__(
        self,
        *,
        title: str,
        template: str,
        key: str,
        default_params: MarkdownParamsT | None = None,
        children: Sequence[Section[SupportsDataclass]] | None = None,
        enabled: Callable[[SupportsDataclass], bool] | None = None,
        tools: Sequence[object] | None = None,
        policies: Sequence[ToolPolicy] | None = None,
        accepts_overrides: bool = True,
        summary: str | None = None,
        visibility: VisibilitySelector = SectionVisibility.FULL,
    ) -> None:
        """Initialize a MarkdownSection.

        Args:
            title: The heading title displayed when the section is rendered.
            template: The markdown template text with ``$placeholder`` or
                ``${placeholder}`` syntax for parameter substitution.
            key: Unique identifier for this section. Must match the pattern
                ``^[a-z0-9][a-z0-9._-]{0,63}$``.
            default_params: Default parameter values used when ``params=None``
                is passed to render methods. Must be an instance of the
                section's parameter type.
            children: Child sections to nest within this section. Children
                are rendered after the parent's body content.
            enabled: Optional predicate that receives params and returns whether
                the section should render. When ``False``, the section is skipped.
            tools: Sequence of :class:`Tool` instances exposed by this section.
                Tools become available to the agent when this section is active.
            policies: Sequence of :class:`ToolPolicy` instances that modify
                tool behavior for this section's scope.
            accepts_overrides: If ``True`` (default), the template body can be
                replaced at render time via :meth:`render_override`.
            summary: Optional abbreviated template text used when the section's
                effective visibility is ``SectionVisibility.SUMMARY``.
            visibility: Controls section visibility. Can be a
                :class:`SectionVisibility` constant or a callable that returns
                one based on params and session state.

        Raises:
            TypeError: If ``default_params`` is provided for a section without
                a declared parameter type.
            TypeError: If ``children`` contains non-Section instances.
            TypeError: If ``tools`` contains non-Tool instances.
        """
        self.template = template
        super().__init__(
            title=title,
            key=key,
            default_params=default_params,
            children=children,
            enabled=enabled,
            tools=tools,
            policies=policies,
            accepts_overrides=accepts_overrides,
            summary=summary,
            visibility=visibility,
        )

    @override
    def render_body(
        self,
        params: SupportsDataclass | None,
        *,
        visibility: SectionVisibility | None = None,
        path: tuple[str, ...] = (),
        session: object = None,
    ) -> str:
        """Render the section body by substituting params into the template.

        When visibility is ``SUMMARY`` and a summary template is defined, the
        summary template is rendered instead of the main template.

        Args:
            params: Parameter dataclass instance for template substitution.
                Falls back to ``default_params`` if ``None``.
            visibility: Effective visibility for rendering. When ``SUMMARY``,
                uses the summary template if available.
            path: Section path (unused in body rendering).
            session: Session context (unused in body rendering).

        Returns:
            The rendered markdown body with all placeholders substituted.

        Raises:
            PromptRenderError: If a required placeholder is missing from params.
        """
        del path, session
        # Use passed visibility directly when provided (already computed by renderer)
        # Fall back to effective_visibility for direct render() calls without renderer
        effective = (
            visibility
            if visibility is not None
            else self.effective_visibility(override=None, params=params)
        )
        template_text = (
            self.summary
            if effective == SectionVisibility.SUMMARY and self.summary is not None
            else self.template
        )
        return self._render_template(template_text, params)

    @override
    def render_override(
        self,
        override_body: str,
        params: SupportsDataclass | None,
        depth: int,
        number: str,
        path: tuple[str, ...] = (),
    ) -> str:
        """Render the section using a custom template instead of the default.

        This method allows runtime customization of the section's body content
        while preserving the heading format and tool examples. The override
        template uses the same ``$placeholder`` syntax as the main template.

        Args:
            override_body: Custom template text to render instead of the
                default ``self.template``. Supports the same placeholder syntax.
            params: Parameter dataclass instance for template substitution.
            depth: Nesting depth (affects heading level: depth 0 = ``##``).
            number: Section number prefix (e.g., "1.2").
            path: Section path tuple for heading annotation.

        Returns:
            Complete rendered markdown including heading, substituted body,
            and any tool examples.

        Raises:
            PromptRenderError: If a required placeholder is missing from params.
        """
        heading = self.format_heading(depth, number, path)
        body = self._render_template(override_body, params)
        rendered_tools = self.render_tool_examples()

        combined_body = body
        if rendered_tools:
            combined_body = f"{body}\n\n{rendered_tools}" if body else rendered_tools

        if combined_body:
            return f"{heading}\n\n{combined_body}"
        return heading

    def _render_template(
        self,
        template_text: str,
        params: SupportsDataclass | None,
    ) -> str:
        """Render template text with parameter substitution.

        Args:
            template_text: The template string to render.
            params: The parameters for substitution.

        Returns:
            The rendered template body (stripped).

        Raises:
            PromptRenderError: When a placeholder is missing.
        """
        template = Template(textwrap.dedent(template_text).strip())
        try:
            normalized_params = self._normalize_params(params)
            rendered_body = template.substitute(normalized_params)
        except KeyError as error:
            missing = error.args[0]
            raise PromptRenderError(
                "Missing placeholder during render.",
                placeholder=str(missing),
            ) from error
        return rendered_body.strip()

    @override
    def placeholder_names(self) -> set[str]:
        """Extract all placeholder names from the section template.

        Scans the template for ``$name`` and ``${name}`` patterns and returns
        the set of placeholder identifiers. This is useful for validation and
        introspection to ensure all required parameters are provided.

        Returns:
            Set of placeholder names found in the template. For a template
            like ``"Hello $name, your role is ${role}"``, returns
            ``{"name", "role"}``.
        """
        template = Template(textwrap.dedent(self.template).strip())
        placeholders: set[str] = set()
        for match in template.pattern.finditer(template.template):
            named = match.group("named")
            if named:
                placeholders.add(named)
                continue
            braced = match.group("braced")
            if braced:
                placeholders.add(braced)
        return placeholders

    @staticmethod
    def _normalize_params(params: SupportsDataclass | None) -> dict[str, Any]:
        if params is None:
            return {}
        if not is_dataclass(params) or isinstance(params, type):
            raise PromptRenderError(
                "Section params must be a dataclass instance.",
                dataclass_type=type(params),
            )

        return {field.name: getattr(params, field.name) for field in fields(params)}

    @override
    def original_body_template(self) -> str:
        """Return the original template text for hashing and comparison.

        This method provides access to the raw template string before any
        parameter substitution, used for content hashing and detecting
        template changes.

        Returns:
            The original template string as provided to ``__init__``.
        """
        return self.template

    @override
    def clone(self, **kwargs: object) -> Self:
        """Create a deep copy of this section and all its children.

        Returns a new MarkdownSection instance with cloned children and
        default_params. The clone is independent of the original - modifications
        to one do not affect the other.

        This method is used by :class:`PromptTemplate` to create isolated
        copies of section trees for different rendering contexts.

        Args:
            **kwargs: Reserved for subclass extensions. Not currently used
                by MarkdownSection.

        Returns:
            A new MarkdownSection instance with the same configuration and
            recursively cloned children.

        Raises:
            TypeError: If any child section does not implement ``clone()``.
        """
        cloned_children: list[Section[SupportsDataclass]] = []
        for child in self.children:
            if not hasattr(child, "clone"):
                raise TypeError(
                    "Section children must implement clone()."
                )  # pragma: no cover
            cloned_children.append(child.clone(**kwargs))

        cloned_default = (
            clone_dataclass(self.default_params)
            if self.default_params is not None
            else None
        )

        cls: type[Any] = type(self)
        clone = cls(
            title=self.title,
            template=self.template,
            key=self.key,
            default_params=cloned_default,
            children=cloned_children,
            enabled=self._enabled,  # ty: ignore[invalid-argument-type]  # callback arity
            tools=self.tools(),
            policies=self.policies(),
            accepts_overrides=self.accepts_overrides,
            summary=self.summary,
            visibility=self.visibility,
        )
        return cast(Self, clone)


__all__ = ["MarkdownSection"]
