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

"""Tier 4 ACK scenarios for tool policy enforcement.

Tool policies are declared on the prompt and enforced by the adapter.
These scenarios verify that adapters correctly deny tool calls when
policies are violated.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from weakincentives.adapters.core import ProviderAdapter
from weakincentives.prompt import (
    MarkdownSection,
    PolicyDecision,
    Prompt,
    PromptTemplate,
    Tool,
    ToolContext,
    ToolResult,
)
from weakincentives.runtime.events.types import ToolInvoked
from weakincentives.runtime.session import Session

from ..adapters import AdapterFixture
from . import make_adapter_ns

pytestmark = pytest.mark.ack_capability("tool_policies")


@dataclass(slots=True)
class WriteParams:
    """Input for the write tool."""

    path: str
    content: str


@dataclass(slots=True, frozen=True)
class WriteResult:
    """Output from the write tool."""

    written: bool


def _build_write_tool(calls: list[str]) -> Tool[WriteParams, WriteResult]:
    """Build a test write tool that records calls."""

    def handler(
        params: WriteParams,
        *,
        context: ToolContext,
    ) -> ToolResult[WriteResult]:
        del context
        calls.append(params.path)
        return ToolResult.ok(WriteResult(written=True), message=f"Wrote {params.path}")

    return Tool[WriteParams, WriteResult](
        name="write_file",
        description="Write content to a file at the given path.",
        handler=handler,
    )


@dataclass(frozen=True)
class AlwaysDenyPolicy:
    """Test policy that denies all write_file calls."""

    @property
    def name(self) -> str:
        return "always_deny"

    def check(
        self,
        tool: Tool[object, object],
        params: object,
        *,
        context: ToolContext,
    ) -> PolicyDecision:
        del params, context
        if tool.name == "write_file":
            return PolicyDecision.deny("Write denied by test policy.")
        return PolicyDecision.allow()

    def on_result(
        self,
        tool: Tool[object, object],
        params: object,
        result: ToolResult[object],
        *,
        context: ToolContext,
    ) -> None:
        del tool, params, result, context


def test_tool_policy_denies_call(
    adapter: ProviderAdapter[object],
    session: Session,
    adapter_fixture: AdapterFixture,
) -> None:
    """Adapter enforces tool policy denial and feeds denial feedback to agent."""
    calls: list[str] = []
    tool = _build_write_tool(calls)

    section = MarkdownSection[WriteParams](
        title="Write Task",
        template=(
            'Call the `write_file` tool with path="output.txt" and '
            'content="hello". If the tool is denied, respond with "denied".'
        ),
        tools=(tool,),
        key="task",
    )

    template = PromptTemplate(
        ns=make_adapter_ns(adapter_fixture.adapter_name),
        key="ack-tool-policy",
        name="ack_tool_policy",
        sections=[section],
        policies=[AlwaysDenyPolicy()],
    )

    prompt = Prompt(template).bind(WriteParams(path="output.txt", content="hello"))

    tool_events: list[ToolInvoked] = []
    session.dispatcher.subscribe(ToolInvoked, tool_events.append)

    response = adapter.evaluate(prompt, session=session)

    # The tool handler should NOT have been called (policy blocked it)
    assert "output.txt" not in calls

    # Agent should have received denial feedback and mentioned it
    assert response.text is not None
    text = response.text.lower()
    denial_indicators = ("denied", "blocked", "policy", "cannot", "not allowed")
    assert any(indicator in text for indicator in denial_indicators)
