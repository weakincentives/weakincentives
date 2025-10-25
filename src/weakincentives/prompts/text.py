from __future__ import annotations

from collections.abc import Sequence
from string import Template
from typing import Any, Callable

import textwrap

from .errors import PromptRenderError
from .section import Section, _ParamsT


class TextSection(Section[_ParamsT]):
    """Render markdown text content using string.Template."""

    def __init__(
        self,
        *,
        title: str,
        body: str,
        params: type[_ParamsT],
        defaults: _ParamsT | None = None,
        children: Sequence[Section[Any]] | None = None,
        enabled: Callable[[_ParamsT], bool] | None = None,
    ) -> None:
        super().__init__(
            title=title,
            params=params,
            defaults=defaults,
            children=children,
            enabled=enabled,
        )
        self.body = body

    def render(self, params: _ParamsT, depth: int) -> str:
        heading_level = "#" * (depth + 2)
        heading = f"{heading_level} {self.title.strip()}"
        template = Template(textwrap.dedent(self.body).strip())
        try:
            rendered_body = template.safe_substitute(vars(params))
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


__all__ = ["TextSection"]
