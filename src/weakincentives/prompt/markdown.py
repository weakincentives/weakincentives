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
    """Render markdown content using :class:`string.Template`."""

    def __init__(
        self,
        *,
        title: str,
        template: str,
        key: str,
        default_params: MarkdownParamsT | None = None,
        children: Sequence[Section[Any]] | None = None,
        enabled: Callable[[SupportsDataclass], bool] | None = None,
        tools: Sequence[object] | None = None,
        policies: Sequence[ToolPolicy] | None = None,
        accepts_overrides: bool = True,
        summary: str | None = None,
        visibility: VisibilitySelector = SectionVisibility.FULL,
    ) -> None:
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
        return self.template

    @override
    def clone(self, **kwargs: object) -> Self:
        cloned_children: list[Section[Any]] = []
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
