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
from typing import Any, Final

from ._types import SupportsDataclass
from .errors import PromptRenderError
from .section import Section


class TextSection[ParamsT: SupportsDataclass](Section[ParamsT]):
    """Render markdown text content using string.Template."""

    def __init__(
        self,
        *,
        title: str,
        body: str,
        key: str | None = None,
        defaults: ParamsT | None = None,
        children: Sequence[object] | None = None,
        enabled: Callable[[ParamsT], bool] | None = None,
        tools: Sequence[object] | None = None,
    ) -> None:
        self.body_template = body
        super().__init__(
            title=title,
            key=key,
            defaults=defaults,
            children=children,
            enabled=enabled,
            tools=tools,
        )

    def render(self, params: ParamsT, depth: int) -> str:
        return self.render_with_body(self.body_template, params, depth)

    def render_with_body(self, body: str, params: ParamsT, depth: int) -> str:
        heading_level = "#" * (depth + 2)
        heading = f"{heading_level} {self.title.strip()}"
        template = Template(textwrap.dedent(body).strip())
        try:
            normalized_params = self._normalize_params(params)
            rendered_body = template.substitute(normalized_params)
        except KeyError as error:  # pragma: no cover - handled at prompt level
            missing = error.args[0]
            raise PromptRenderError(
                "Missing placeholder during render.",
                placeholder=str(missing),
            ) from error
        if rendered_body:
            return f"{heading}\n\n{rendered_body.strip()}"
        return heading

    def placeholder_names(self) -> set[str]:
        template = Template(textwrap.dedent(self.body_template).strip())
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
    def _normalize_params(params: ParamsT) -> dict[str, Any]:
        if not is_dataclass(params) or isinstance(params, type):
            raise PromptRenderError(
                "Section params must be a dataclass instance.",
                dataclass_type=type(params),
            )

        return {field.name: getattr(params, field.name) for field in fields(params)}

    def original_body_template(self) -> str:
        return self.body_template


__all__: Final[list[str]] = ["TextSection"]
