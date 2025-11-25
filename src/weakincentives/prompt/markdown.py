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
from typing import Any, Self, TypeVar, cast, override

from ..serde import clone as clone_dataclass
from ._types import SupportsDataclass
from .errors import PromptRenderError
from .section import Section

MarkdownParamsT = TypeVar(
    "MarkdownParamsT",
    bound=SupportsDataclass,
    covariant=True,
)


class MarkdownSection(Section[MarkdownParamsT]):
    """Render markdown content using :class:`string.Template`."""

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
        accepts_overrides: bool = True,
    ) -> None:
        self.template = template
        super().__init__(
            title=title,
            key=key,
            default_params=default_params,
            children=children,
            enabled=enabled,
            tools=tools,
            accepts_overrides=accepts_overrides,
        )

    @override
    def render(self, params: SupportsDataclass | None, depth: int, number: str) -> str:
        return self.render_with_template(self.template, params, depth, number)

    def render_with_template(
        self,
        template_text: str,
        params: SupportsDataclass | None,
        depth: int,
        number: str,
    ) -> str:
        heading_level = "#" * (depth + 2)
        normalized_number = number.rstrip(".")
        heading = f"{heading_level} {normalized_number}. {self.title.strip()}"
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
        if rendered_body:
            return f"{heading}\n\n{rendered_body.strip()}"
        return heading

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
            enabled=self._enabled,
            tools=self.tools(),
            accepts_overrides=self.accepts_overrides,
        )
        return cast(Self, clone)


__all__ = ["MarkdownSection"]
