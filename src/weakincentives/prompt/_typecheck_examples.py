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

from __future__ import annotations  # pragma: no cover

from typing import TYPE_CHECKING  # pragma: no cover

if TYPE_CHECKING:  # pragma: no cover
    from dataclasses import dataclass
    from typing import Literal, assert_type

    from ._types import SupportsDataclass
    from .markdown import MarkdownSection
    from .tool import Tool, ToolExample, ToolSpec

    # Section type checking examples
    _parameterless_section = MarkdownSection[SupportsDataclass](
        title="Example",
        template="static content",
        key="example-section",
    )
    _ = assert_type(_parameterless_section, MarkdownSection[SupportsDataclass])

    # ToolSpec protocol type checking examples

    @dataclass(slots=True, frozen=True)
    class _ExampleParams:
        value: str

    @dataclass(slots=True, frozen=True)
    class _ExampleResult:
        output: str

    # Tool implements ToolSpec - verify structural subtyping
    def _get_tool_instance() -> Tool[_ExampleParams, _ExampleResult]: ...

    _tool_instance = _get_tool_instance()

    # Tool is assignable to ToolSpec (structural subtyping)
    # The assignment itself verifies Tool satisfies ToolSpec
    _tool_as_spec: ToolSpec[_ExampleParams, _ExampleResult] = _tool_instance

    # ToolSpec properties are accessible
    _spec_name: str = _tool_as_spec.name
    _spec_description: str = _tool_as_spec.description
    _spec_params_type: type[_ExampleParams] = _tool_as_spec.params_type
    _spec_result_type: type[SupportsDataclass] | type[None] = _tool_as_spec.result_type
    _spec_result_container: Literal["object", "array"] = _tool_as_spec.result_container
    _spec_examples: tuple[ToolExample[_ExampleParams, _ExampleResult], ...] = (
        _tool_as_spec.examples
    )
    _spec_accepts_overrides: bool = _tool_as_spec.accepts_overrides

    # Custom class implementing ToolSpec protocol (structural subtyping)
    @dataclass(slots=True, frozen=True)
    class _CustomToolSpec:
        """Custom tool implementation that satisfies ToolSpec protocol."""

        name: str
        description: str
        params_type: type[_ExampleParams]
        result_type: type[_ExampleResult]
        result_container: Literal["object", "array"]
        examples: tuple[ToolExample[_ExampleParams, _ExampleResult], ...]
        accepts_overrides: bool
        handler: None = None  # No handler for documentation-only tools

    _custom_tool = _CustomToolSpec(
        name="custom_tool",
        description="A custom tool",
        params_type=_ExampleParams,
        result_type=_ExampleResult,
        result_container="object",
        examples=(),
        accepts_overrides=True,
        handler=None,
    )

    # Custom tool should be assignable to ToolSpec
    # The assignment itself verifies _CustomToolSpec satisfies ToolSpec
    _custom_as_spec: ToolSpec[_ExampleParams, _ExampleResult] = _custom_tool

    # Section can accept ToolSpec implementations
    _section_with_tools = MarkdownSection[SupportsDataclass](
        title="Tools Section",
        template="content",
        key="tools-section",
        tools=[_tool_as_spec, _custom_as_spec],  # Both satisfy ToolSpec
    )
    _ = assert_type(_section_with_tools, MarkdownSection[SupportsDataclass])
