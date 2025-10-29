from __future__ import annotations

import textwrap
from collections.abc import Callable, Sequence
from dataclasses import fields, is_dataclass
from string import Template
from typing import Any

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
        defaults: ParamsT | None = None,
        children: Sequence[object] | None = None,
        enabled: Callable[[ParamsT], bool] | None = None,
        tools: Sequence[object] | None = None,
    ) -> None:
        super().__init__(
            title=title,
            defaults=defaults,
            children=children,
            enabled=enabled,
            tools=tools,
        )
        self.body = body

    def render(self, params: ParamsT, depth: int) -> str:
        heading_level = "#" * (depth + 2)
        heading = f"{heading_level} {self.title.strip()}"
        template = Template(textwrap.dedent(self.body).strip())
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
        template = Template(textwrap.dedent(self.body).strip())
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


__all__ = ["TextSection"]
