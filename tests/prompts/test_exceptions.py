from __future__ import annotations

from dataclasses import dataclass

import pytest

from weakincentives.prompts import PromptRenderError, PromptValidationError


@dataclass
class FakeParams:
    value: str = "example"


@pytest.mark.parametrize("exception_cls", [PromptValidationError, PromptRenderError])
def test_prompt_exceptions_capture_context(exception_cls):
    context = {
        "section_path": ("Root", "Child"),
        "dataclass_type": FakeParams,
        "placeholder": "value",
    }

    exc = exception_cls("boom", **context)

    assert isinstance(exc, Exception)
    assert exc.message == "boom"
    assert exc.args[0] == "boom"
    assert exc.section_path == context["section_path"]
    assert exc.dataclass_type is FakeParams
    assert exc.placeholder == "value"


def test_prompt_exceptions_default_context_optional():
    exc = PromptValidationError(
        "missing", section_path=None, dataclass_type=None, placeholder=None
    )

    assert exc.section_path == ()
    assert exc.dataclass_type is None
    assert exc.placeholder is None
