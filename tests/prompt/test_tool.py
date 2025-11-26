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

from dataclasses import dataclass

import pytest

from weakincentives.prompt import (
    PromptValidationError,
    Tool,
    ToolExample,
)


@dataclass
class ExampleParams:
    query: str


@dataclass
class ExampleResult:
    value: str


@dataclass
class OtherParams:
    value: int


def test_tool_examples_are_preserved() -> None:
    example = ToolExample[ExampleParams, ExampleResult](
        description="simple lookup",
        input=ExampleParams(query="widgets"),
        output=ExampleResult(value="result"),
    )

    tool = Tool[ExampleParams, ExampleResult](
        name="lookup",
        description="Lookup information.",
        handler=None,
        examples=(example,),
    )

    assert tool.examples == (example,)


def test_tool_example_requires_ascii_description() -> None:
    example = ToolExample[ExampleParams, ExampleResult](
        description="emoji 😊",
        input=ExampleParams(query="widgets"),
        output=ExampleResult(value="result"),
    )

    with pytest.raises(PromptValidationError):
        Tool[ExampleParams, ExampleResult](
            name="lookup",
            description="Lookup information.",
            handler=None,
            examples=(example,),
        )


def test_tool_example_input_must_match_params_type() -> None:
    example = ToolExample[ExampleParams, ExampleResult](
        description="mismatch",
        input=OtherParams(value=1),  # type: ignore[arg-type]
        output=ExampleResult(value="result"),
    )

    with pytest.raises(PromptValidationError):
        Tool[ExampleParams, ExampleResult](
            name="lookup",
            description="Lookup information.",
            handler=None,
            examples=(example,),
        )


def test_tool_example_output_must_match_sequence_when_result_is_array() -> None:
    valid_example = ToolExample[ExampleParams, list[ExampleResult]](
        description="sequence output",
        input=ExampleParams(query="widgets"),
        output=[ExampleResult(value="first"), ExampleResult(value="second")],
    )

    tool = Tool[ExampleParams, list[ExampleResult]](
        name="batch_lookup",
        description="Lookup multiple results.",
        handler=None,
        examples=(valid_example,),
    )

    assert tool.examples == (valid_example,)

    invalid_example = ToolExample[ExampleParams, list[ExampleResult]](
        description="not a sequence",
        input=ExampleParams(query="widgets"),
        output=ExampleResult(value="lonely"),  # type: ignore[arg-type]
    )

    with pytest.raises(PromptValidationError):
        Tool[ExampleParams, list[ExampleResult]](
            name="batch_lookup",
            description="Lookup multiple results.",
            handler=None,
            examples=(invalid_example,),
        )


def test_tool_example_rejects_blank_description() -> None:
    example = ToolExample[ExampleParams, ExampleResult](
        description="   ",
        input=ExampleParams(query="widgets"),
        output=ExampleResult(value="result"),
    )

    with pytest.raises(PromptValidationError):
        Tool[ExampleParams, ExampleResult](
            name="lookup",
            description="Lookup information.",
            handler=None,
            examples=(example,),
        )


def test_tool_example_input_requires_dataclass_instance() -> None:
    example = ToolExample[ExampleParams, ExampleResult](
        description="non dataclass",
        input={"query": "widgets"},  # type: ignore[arg-type]
        output=ExampleResult(value="result"),
    )

    with pytest.raises(PromptValidationError):
        Tool[ExampleParams, ExampleResult](
            name="lookup",
            description="Lookup information.",
            handler=None,
            examples=(example,),
        )


def test_tool_example_output_validates_sequence_items() -> None:
    example = ToolExample[ExampleParams, list[ExampleResult]](
        description="wrong items",
        input=ExampleParams(query="widgets"),
        output=[ExampleParams(query="widgets")],  # type: ignore[list-item]
    )

    with pytest.raises(PromptValidationError):
        Tool[ExampleParams, list[ExampleResult]](
            name="batch_lookup",
            description="Lookup multiple results.",
            handler=None,
            examples=(example,),
        )


def test_tool_example_output_requires_dataclass_instance_for_object_results() -> None:
    example = ToolExample[ExampleParams, ExampleResult](
        description="wrong output type",
        input=ExampleParams(query="widgets"),
        output={"value": "result"},  # type: ignore[arg-type]
    )

    with pytest.raises(PromptValidationError):
        Tool[ExampleParams, ExampleResult](
            name="lookup",
            description="Lookup information.",
            handler=None,
            examples=(example,),
        )


def test_tool_examples_must_be_tool_example_instances() -> None:
    with pytest.raises(PromptValidationError):
        Tool[ExampleParams, ExampleResult](
            name="lookup",
            description="Lookup information.",
            handler=None,
            examples=("not-an-example",),  # type: ignore[arg-type]
        )
