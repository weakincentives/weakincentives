"""Prompt module scaffolding."""

from __future__ import annotations

from ._types import SupportsDataclass
from .errors import (
    PromptError,
    PromptRenderError,
    PromptValidationError,
    SectionPath,
)
from .prompt import Prompt, PromptSectionNode, RenderedPrompt
from .section import Section
from .structured import OutputParseError, parse_output
from .text import TextSection
from .tool import Tool, ToolResult

__all__ = [
    "Prompt",
    "RenderedPrompt",
    "PromptSectionNode",
    "PromptError",
    "PromptRenderError",
    "PromptValidationError",
    "Section",
    "SectionPath",
    "SupportsDataclass",
    "TextSection",
    "Tool",
    "ToolResult",
    "OutputParseError",
    "parse_output",
]
