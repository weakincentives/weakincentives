from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from .text import TextSection

__all__ = ["ResponseFormatParams", "ResponseFormatSection"]


@dataclass(slots=True)
class ResponseFormatParams:
    """Parameter payload for the auto-generated response format section."""

    article: Literal["a", "an"]
    container: Literal["object", "array"]
    extra_clause: str


_RESPONSE_FORMAT_BODY = """Return ONLY a single fenced JSON code block. Do not include any text
before or after the block.

The top-level JSON value MUST be ${article} ${container} that matches the fields
of the expected schema${extra_clause}"""


class ResponseFormatSection(TextSection[ResponseFormatParams]):
    """Internal section that appends JSON-only response instructions."""

    def __init__(
        self,
        *,
        params: ResponseFormatParams,
        enabled: Callable[[ResponseFormatParams], bool] | None = None,
    ) -> None:
        super().__init__(
            title="Response Format",
            body=_RESPONSE_FORMAT_BODY,
            defaults=params,
            enabled=enabled,
        )
