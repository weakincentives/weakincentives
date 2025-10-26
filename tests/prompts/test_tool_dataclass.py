from __future__ import annotations

from dataclasses import dataclass

import pytest

from weakincentives.prompts.errors import PromptValidationError
from weakincentives.prompts.tool import Tool


@dataclass
class ExampleParams:
    message: str


def test_tool_construction_strips_description_whitespace() -> None:
    tool = Tool(
        name="lookup_entity",
        description="  Fetch structured entity info.  ",
        params=ExampleParams,
    )

    assert tool.name == "lookup_entity"
    assert tool.description == "Fetch structured entity info."
    assert tool.params is ExampleParams
    assert tool.handler is None


@pytest.mark.parametrize(
    "name",
    [
        "",
        "Lookup",
        "lookup-entity",
        "lookup entity",
        "lookup.entity",
        "lookup/entity",
        "lookup$entity",
        "工具",
        "a" * 65,
        " lookup ",
    ],
)
def test_tool_rejects_invalid_names(name: str) -> None:
    with pytest.raises(PromptValidationError) as error_info:
        Tool(name=name, description="valid description", params=ExampleParams)

    error = error_info.value
    assert isinstance(error, PromptValidationError)
    assert error.dataclass_type is ExampleParams
    assert error.placeholder == name.strip()


@pytest.mark.parametrize(
    "description",
    [
        "",
        "   ",
        "a" * 201,
        "déjà vu",
    ],
)
def test_tool_rejects_invalid_descriptions(description: str) -> None:
    with pytest.raises(PromptValidationError) as error_info:
        Tool(name="lookup_entity", description=description, params=ExampleParams)

    error = error_info.value
    assert isinstance(error, PromptValidationError)
    assert error.dataclass_type is ExampleParams
    assert error.placeholder == "description"


def test_tool_requires_dataclass_params() -> None:
    with pytest.raises(PromptValidationError) as error_info:
        Tool(name="lookup_entity", description="Valid description", params=str)  # type: ignore[arg-type]

    error = error_info.value
    assert isinstance(error, PromptValidationError)
    assert error.dataclass_type is str
    assert error.placeholder == "params"
