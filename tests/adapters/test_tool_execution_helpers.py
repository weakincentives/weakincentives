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

import json
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import pytest

try:
    from tests.adapters._test_stubs import DummyToolCall, ToolParams, ToolPayload
except ModuleNotFoundError:  # pragma: no cover - fallback for direct invocation
    from ._test_stubs import DummyToolCall, ToolParams, ToolPayload

from tests.helpers import FrozenUtcNow
from weakincentives import DeadlineExceededError
from weakincentives.adapters.core import (
    PROMPT_EVALUATION_PHASE_TOOL,
    PromptEvaluationError,
)
from weakincentives.adapters.tool_executor import (
    RejectedToolParams,
    ToolExecutionContext,
    ToolExecutionOutcome,
    tool_execution,
)
from weakincentives.adapters.utilities import (
    format_dispatch_failures,
    parse_tool_arguments,
)
from weakincentives.deadlines import Deadline
from weakincentives.prompt import (
    MarkdownSection,
    Prompt,
    PromptTemplate,
    Tool,
    ToolContext,
    ToolHandler,
    ToolResult,
)
from weakincentives.runtime.events import InProcessDispatcher
from weakincentives.runtime.session import Session, SessionProtocol
from weakincentives.types import SupportsDataclassOrNone, SupportsToolResult


def _build_prompt(tool: Tool[ToolParams, ToolPayload]) -> PromptTemplate[ToolPayload]:
    return PromptTemplate(
        ns="tests/adapters/tool-execution-helpers",
        key="tool-execution-helpers",
        name="test",
        sections=(
            MarkdownSection[ToolParams](
                title="Task",
                key="task",
                template="Look up ${query}",
                tools=[tool],
            ),
        ),
    )


def _base_context(
    tool: Tool[ToolParams, ToolPayload],
    *,
    deadline: Deadline | None = None,
    session: SessionProtocol | None = None,
) -> ToolExecutionContext:
    bus = InProcessDispatcher()
    prompt_template = _build_prompt(tool)
    prompt: Prompt[ToolPayload] = Prompt(prompt_template)
    effective_session = session or Session(bus=bus)
    # Enter prompt context for resource access
    prompt.__enter__()
    return ToolExecutionContext(
        adapter_name="adapter",
        adapter=cast(Any, object()),
        prompt=prompt,
        rendered_prompt=None,
        tool_registry=cast(
            Mapping[str, Tool[SupportsDataclassOrNone, SupportsToolResult]],
            {tool.name: cast(Tool[SupportsDataclassOrNone, SupportsToolResult], tool)},
        ),
        session=effective_session,
        prompt_name=cast(str, prompt.name),
        parse_arguments=parse_tool_arguments,
        format_dispatch_failures=format_dispatch_failures,
        deadline=deadline,
        provider_payload={},
    )


def _build_tool(
    handler: ToolHandler[ToolParams, ToolPayload],
) -> Tool[ToolParams, ToolPayload]:
    return Tool[ToolParams, ToolPayload](
        name="search_notes",
        description="Search stored notes.",
        handler=handler,
    )


def _tool_call(arguments: object) -> DummyToolCall:
    return DummyToolCall(
        call_id="call-id",
        name="search_notes",
        arguments=json.dumps(arguments),
    )


def test_tool_execution_success_path() -> None:
    def handler(params: ToolParams, *, context: ToolContext) -> ToolResult[ToolPayload]:
        assert context.session is not None
        return ToolResult.ok(ToolPayload(answer=params.query), message="done")

    tool = _build_tool(cast(ToolHandler[ToolParams, ToolPayload], handler))
    tool_call = _tool_call({"query": "policies"})
    context = _base_context(tool)

    with tool_execution(context=context, tool_call=tool_call) as outcome:
        assert isinstance(outcome, ToolExecutionOutcome)
        assert outcome.result.success is True
        assert outcome.result.value == ToolPayload(answer="policies")
        assert outcome.params == ToolParams(query="policies")


def test_tool_execution_records_validation_failure() -> None:
    def handler(params: ToolParams, *, context: ToolContext) -> ToolResult[ToolPayload]:
        del params, context
        return ToolResult.ok(ToolPayload(answer="noop"), message="ok")

    tool = _build_tool(cast(ToolHandler[ToolParams, ToolPayload], handler))
    tool_call = _tool_call({"query": "policies", "extra": True})
    context = _base_context(tool)

    with tool_execution(context=context, tool_call=tool_call) as outcome:
        assert isinstance(outcome.params, RejectedToolParams)
        assert outcome.params.raw_arguments == {"query": "policies", "extra": True}
        assert outcome.result.success is False
        assert "Tool validation failed" in outcome.result.message


def test_tool_execution_raises_on_expired_deadline(
    frozen_utcnow: FrozenUtcNow,
) -> None:
    def handler(params: ToolParams, *, context: ToolContext) -> ToolResult[ToolPayload]:
        del params, context
        return ToolResult.ok(ToolPayload(answer="x"), message="should not run")

    tool = _build_tool(cast(ToolHandler[ToolParams, ToolPayload], handler))
    anchor = datetime.now(UTC)
    frozen_utcnow.set(anchor)
    expired_deadline = Deadline(anchor + timedelta(seconds=2))
    frozen_utcnow.advance(timedelta(seconds=5))
    context = _base_context(tool, deadline=expired_deadline)
    tool_call = _tool_call({"query": "policies"})

    with (
        pytest.raises(PromptEvaluationError) as excinfo,
        tool_execution(context=context, tool_call=tool_call),
    ):
        pytest.fail("tool_execution should raise before yielding")

    error = excinfo.value
    assert error.phase == PROMPT_EVALUATION_PHASE_TOOL
    assert error.provider_payload == {
        "deadline_expires_at": expired_deadline.expires_at.isoformat()
    }


def test_tool_execution_wraps_handler_deadline_exceptions() -> None:
    def handler(params: ToolParams, *, context: ToolContext) -> ToolResult[ToolPayload]:
        del params, context
        raise DeadlineExceededError("deadline hit")

    tool = _build_tool(cast(ToolHandler[ToolParams, ToolPayload], handler))
    deadline = Deadline(datetime.now(UTC) + timedelta(seconds=10))
    context = _base_context(tool, deadline=deadline)
    tool_call = _tool_call({"query": "policies"})

    with (
        pytest.raises(PromptEvaluationError) as excinfo,
        tool_execution(context=context, tool_call=tool_call),
    ):
        pytest.fail("tool_execution should raise before yielding")

    error = excinfo.value
    assert error.phase == PROMPT_EVALUATION_PHASE_TOOL
    assert error.provider_payload == {
        "deadline_expires_at": deadline.expires_at.isoformat()
    }


def test_tool_execution_converts_unexpected_exceptions() -> None:
    def handler(params: ToolParams, *, context: ToolContext) -> ToolResult[ToolPayload]:
        del params, context
        raise RuntimeError("boom")

    tool = _build_tool(cast(ToolHandler[ToolParams, ToolPayload], handler))
    context = _base_context(tool)
    tool_call = _tool_call({"query": "policies"})

    with tool_execution(context=context, tool_call=tool_call) as outcome:
        assert outcome.result.success is False
        assert outcome.result.message == "Tool 'search_notes' execution failed: boom"
        assert outcome.params == ToolParams(query="policies")


def test_tool_execution_handles_type_error_as_possible_signature_mismatch() -> None:
    """TypeError during execution is treated as possible signature mismatch.

    While pyright catches most signature issues at development time,
    TypeErrors can still occur at runtime and should be handled gracefully.
    """

    def handler(params: ToolParams, *, context: ToolContext) -> ToolResult[ToolPayload]:
        del params, context
        raise TypeError("handler() missing 1 required keyword-only argument: 'context'")

    tool = _build_tool(cast(ToolHandler[ToolParams, ToolPayload], handler))
    context = _base_context(tool)
    tool_call = _tool_call({"query": "policies"})

    with tool_execution(context=context, tool_call=tool_call) as outcome:
        assert outcome.result.success is False
        assert "TypeError" in outcome.result.message
        assert "search_notes" in outcome.result.message
