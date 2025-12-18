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

"""Tests for ToolRunner unified tool execution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, cast

import pytest

try:
    from tests.adapters._test_stubs import (
        DummyToolCall,
        ToolParams,
        ToolPayload,
    )
except ModuleNotFoundError:  # pragma: no cover
    from ._test_stubs import DummyToolCall, ToolParams, ToolPayload

from weakincentives.adapters.tool_runner import ToolRunner
from weakincentives.contrib.tools.filesystem import Filesystem, InMemoryFilesystem
from weakincentives.errors import DeadlineExceededError
from weakincentives.prompt import (
    MarkdownSection,
    Prompt,
    PromptTemplate,
    Tool,
    ToolContext,
)
from weakincentives.prompt.errors import VisibilityExpansionRequired
from weakincentives.prompt.progressive_disclosure import SectionVisibility
from weakincentives.prompt.tool import ResourceRegistry, ToolResult
from weakincentives.runtime.execution_state import ExecutionState
from weakincentives.runtime.session import Session
from weakincentives.types import SupportsDataclassOrNone, SupportsToolResult


def _make_visibility_expansion() -> VisibilityExpansionRequired:
    """Create a VisibilityExpansionRequired exception for testing."""
    return VisibilityExpansionRequired(
        message="Need expansion",
        requested_overrides={("task",): SectionVisibility.FULL},
        reason="Model requested expansion",
        section_keys=("task",),
    )


# --- Test Data ---


@dataclass(frozen=True)
class SamplePlan:
    """Test dataclass for session state."""

    objective: str


def _success_handler(
    params: ToolParams, *, context: ToolContext
) -> ToolResult[ToolPayload]:
    """Handler that succeeds."""
    return ToolResult(message="ok", value=ToolPayload(answer=params.query))


def _failure_handler(
    params: ToolParams, *, context: ToolContext
) -> ToolResult[ToolPayload]:
    """Handler that returns failure."""
    return ToolResult(message="failed", value=None, success=False)


def _exception_handler(
    params: ToolParams, *, context: ToolContext
) -> ToolResult[ToolPayload]:
    """Handler that raises an exception."""
    msg = "Something went wrong"
    raise RuntimeError(msg)


def _visibility_expansion_handler(
    params: ToolParams, *, context: ToolContext
) -> ToolResult[ToolPayload]:
    """Handler that raises VisibilityExpansionRequired."""
    raise _make_visibility_expansion()


def _mutating_handler(
    params: ToolParams, *, context: ToolContext
) -> ToolResult[ToolPayload]:
    """Handler that mutates filesystem and session."""
    fs = context.filesystem
    if fs is not None:
        fs.write("modified.txt", "mutated content")
    # Note: Session mutations would happen through broadcast
    return ToolResult(message="mutated", value=ToolPayload(answer="done"))


def _deadline_handler(
    params: ToolParams, *, context: ToolContext
) -> ToolResult[ToolPayload]:
    """Handler that raises DeadlineExceededError."""
    raise DeadlineExceededError("Deadline exceeded")


# --- Fixtures ---


def _build_tool_registry(
    tool: Tool[Any, Any],
) -> dict[str, Tool[SupportsDataclassOrNone, SupportsToolResult]]:
    return cast(
        dict[str, Tool[SupportsDataclassOrNone, SupportsToolResult]],
        {tool.name: cast(Tool[SupportsDataclassOrNone, SupportsToolResult], tool)},
    )


def _build_tool_context(
    session: Session,
    resources: ResourceRegistry | None = None,
) -> ToolContext:
    prompt_template = PromptTemplate[ToolPayload](
        ns="tests/tool-runner",
        key="tool-runner",
        name="test",
        sections=(
            MarkdownSection[ToolParams](
                title="Task",
                key="task",
                template="Look up ${query}",
                tools=[],
            ),
        ),
    )
    prompt = Prompt(prompt_template)
    return ToolContext(
        prompt=prompt,
        rendered_prompt=None,
        adapter=cast(Any, object()),
        session=session,
        deadline=None,
        resources=resources or ResourceRegistry(),
    )


def _make_tool_call(name: str, arguments: dict[str, Any]) -> DummyToolCall:
    return DummyToolCall(
        call_id="call_123",
        name=name,
        arguments=json.dumps(arguments),
    )


# --- ToolRunner Tests ---


class TestToolRunnerBasic:
    def test_execute_successful_tool(self) -> None:
        """Successful tool execution returns result."""
        tool = Tool[ToolParams, ToolPayload](
            name="test_tool",
            description="A test tool",
            handler=_success_handler,
        )
        session = Session()
        state = ExecutionState(session=session)

        runner = ToolRunner(
            execution_state=state,
            tool_registry=_build_tool_registry(tool),
            prompt_name="test",
        )
        context = _build_tool_context(session)
        tool_call = _make_tool_call("test_tool", {"query": "hello"})

        result = runner.execute(tool_call, context=context)

        assert result.success
        assert result.value is not None
        assert result.value.answer == "hello"

    def test_execute_unknown_tool(self) -> None:
        """Unknown tool returns failure result."""
        session = Session()
        state = ExecutionState(session=session)

        runner = ToolRunner(
            execution_state=state,
            tool_registry={},
            prompt_name="test",
        )
        context = _build_tool_context(session)
        tool_call = _make_tool_call("unknown_tool", {"query": "hello"})

        result = runner.execute(tool_call, context=context)

        assert not result.success
        assert "Unknown tool" in result.message

    def test_execute_tool_without_handler(self) -> None:
        """Tool without handler returns failure result."""
        tool = Tool[ToolParams, ToolPayload](
            name="no_handler",
            description="A tool without handler",
            handler=None,
        )
        session = Session()
        state = ExecutionState(session=session)

        runner = ToolRunner(
            execution_state=state,
            tool_registry=_build_tool_registry(tool),
            prompt_name="test",
        )
        context = _build_tool_context(session)
        tool_call = _make_tool_call("no_handler", {"query": "hello"})

        result = runner.execute(tool_call, context=context)

        assert not result.success
        assert "does not have a handler" in result.message


class TestToolRunnerStateRestoration:
    def test_restore_on_failure_result(self) -> None:
        """State is restored when tool returns success=False."""
        tool = Tool[ToolParams, ToolPayload](
            name="failing_tool",
            description="A failing tool",
            handler=_failure_handler,
        )
        session = Session()
        fs = InMemoryFilesystem()
        fs.write("original.txt", "original")
        resources = ResourceRegistry.build({Filesystem: fs})
        state = ExecutionState(session=session, resources=resources)

        def mutating_failure_handler(
            params: ToolParams, *, context: ToolContext
        ) -> ToolResult[ToolPayload]:
            fs.write("original.txt", "modified")
            return ToolResult(message="failed", value=None, success=False)

        tool = Tool[ToolParams, ToolPayload](
            name="failing_tool",
            description="A failing tool",
            handler=mutating_failure_handler,
        )

        runner = ToolRunner(
            execution_state=state,
            tool_registry=_build_tool_registry(tool),
            prompt_name="test",
        )
        context = _build_tool_context(session, resources)
        tool_call = _make_tool_call("failing_tool", {"query": "hello"})

        result = runner.execute(tool_call, context=context)

        assert not result.success
        # State should be restored
        assert fs.read("original.txt").content == "original"

    def test_restore_on_exception(self) -> None:
        """State is restored when handler raises exception."""
        tool = Tool[ToolParams, ToolPayload](
            name="exception_tool",
            description="A tool that raises",
            handler=_exception_handler,
        )
        session = Session()
        fs = InMemoryFilesystem()
        fs.write("data.txt", "original")
        resources = ResourceRegistry.build({Filesystem: fs})
        state = ExecutionState(session=session, resources=resources)

        # Pre-mutate in a wrapper that raises
        def exception_with_mutation(
            params: ToolParams, *, context: ToolContext
        ) -> ToolResult[ToolPayload]:
            fs.write("data.txt", "modified before exception")
            raise RuntimeError("boom")

        tool = Tool[ToolParams, ToolPayload](
            name="exception_tool",
            description="A tool that raises",
            handler=exception_with_mutation,
        )

        runner = ToolRunner(
            execution_state=state,
            tool_registry=_build_tool_registry(tool),
            prompt_name="test",
        )
        context = _build_tool_context(session, resources)
        tool_call = _make_tool_call("exception_tool", {"query": "hello"})

        result = runner.execute(tool_call, context=context)

        assert not result.success
        assert "execution failed" in result.message
        # State should be restored
        assert fs.read("data.txt").content == "original"

    def test_restore_on_visibility_expansion(self) -> None:
        """State is restored when VisibilityExpansionRequired is raised."""
        tool = Tool[ToolParams, ToolPayload](
            name="visibility_tool",
            description="A tool that needs visibility expansion",
            handler=_visibility_expansion_handler,
        )
        session = Session()
        fs = InMemoryFilesystem()
        fs.write("file.txt", "original")
        resources = ResourceRegistry.build({Filesystem: fs})
        state = ExecutionState(session=session, resources=resources)

        def visibility_with_mutation(
            params: ToolParams, *, context: ToolContext
        ) -> ToolResult[ToolPayload]:
            fs.write("file.txt", "modified")
            raise _make_visibility_expansion()

        tool = Tool[ToolParams, ToolPayload](
            name="visibility_tool",
            description="A tool that needs visibility expansion",
            handler=visibility_with_mutation,
        )

        runner = ToolRunner(
            execution_state=state,
            tool_registry=_build_tool_registry(tool),
            prompt_name="test",
        )
        context = _build_tool_context(session, resources)
        tool_call = _make_tool_call("visibility_tool", {"query": "hello"})

        with pytest.raises(VisibilityExpansionRequired):
            runner.execute(tool_call, context=context)

        # State should be restored
        assert fs.read("file.txt").content == "original"

    def test_no_restore_on_success(self) -> None:
        """State changes are preserved on successful execution."""
        tool = Tool[ToolParams, ToolPayload](
            name="mutating_tool",
            description="A tool that mutates state",
            handler=_mutating_handler,
        )
        session = Session()
        fs = InMemoryFilesystem()
        fs.write("original.txt", "original")
        resources = ResourceRegistry.build({Filesystem: fs})
        state = ExecutionState(session=session, resources=resources)

        runner = ToolRunner(
            execution_state=state,
            tool_registry=_build_tool_registry(tool),
            prompt_name="test",
        )
        context = _build_tool_context(session, resources)
        tool_call = _make_tool_call("mutating_tool", {"query": "hello"})

        result = runner.execute(tool_call, context=context)

        assert result.success
        # Mutations should persist
        assert fs.read("modified.txt").content == "mutated content"


class TestToolRunnerValidation:
    def test_invalid_json_arguments(self) -> None:
        """Invalid JSON arguments cause validation error."""
        tool = Tool[ToolParams, ToolPayload](
            name="test_tool",
            description="A test tool",
            handler=_success_handler,
        )
        session = Session()
        state = ExecutionState(session=session)

        runner = ToolRunner(
            execution_state=state,
            tool_registry=_build_tool_registry(tool),
            prompt_name="test",
        )
        context = _build_tool_context(session)
        # Create tool call with invalid JSON
        tool_call = DummyToolCall(
            call_id="call_123",
            name="test_tool",
            arguments="not-valid-json",
        )

        result = runner.execute(tool_call, context=context)

        assert not result.success
        assert "execution failed" in result.message

    def test_non_object_arguments(self) -> None:
        """Non-object JSON arguments cause validation error."""
        tool = Tool[ToolParams, ToolPayload](
            name="test_tool",
            description="A test tool",
            handler=_success_handler,
        )
        session = Session()
        state = ExecutionState(session=session)

        runner = ToolRunner(
            execution_state=state,
            tool_registry=_build_tool_registry(tool),
            prompt_name="test",
        )
        context = _build_tool_context(session)
        # Create tool call with array instead of object
        tool_call = DummyToolCall(
            call_id="call_123",
            name="test_tool",
            arguments='["array", "not", "object"]',
        )

        result = runner.execute(tool_call, context=context)

        assert not result.success


class TestToolRunnerDeadline:
    def test_deadline_exceeded_raises_error(self) -> None:
        """DeadlineExceededError is converted to PromptEvaluationError."""
        tool = Tool[ToolParams, ToolPayload](
            name="deadline_tool",
            description="A tool that exceeds deadline",
            handler=_deadline_handler,
        )
        session = Session()
        state = ExecutionState(session=session)

        runner = ToolRunner(
            execution_state=state,
            tool_registry=_build_tool_registry(tool),
            prompt_name="test",
        )
        context = _build_tool_context(session)
        tool_call = _make_tool_call("deadline_tool", {"query": "hello"})

        from weakincentives.adapters.core import PromptEvaluationError

        with pytest.raises(PromptEvaluationError) as exc_info:
            runner.execute(tool_call, context=context)

        assert "Deadline exceeded" in str(exc_info.value)


class TestToolRunnerAcceptanceCriteria:
    def test_tool_failure_does_not_change_state(self) -> None:
        """Acceptance criteria: Tool failure does not change state."""
        # Setup with initial state
        session = Session()
        session[SamplePlan].seed([SamplePlan(objective="test")])
        fs = InMemoryFilesystem()
        fs.write("file.txt", "original")
        resources = ResourceRegistry.build({Filesystem: fs})
        state = ExecutionState(session=session, resources=resources)

        # Create tool that mutates and fails
        def mutate_and_fail(
            params: ToolParams, *, context: ToolContext
        ) -> ToolResult[ToolPayload]:
            if context.filesystem is not None:
                context.filesystem.write("file.txt", "modified")
            return ToolResult(message="failed", value=None, success=False)

        tool = Tool[ToolParams, ToolPayload](
            name="mutating_fail",
            description="Mutates then fails",
            handler=mutate_and_fail,
        )

        runner = ToolRunner(
            execution_state=state,
            tool_registry=_build_tool_registry(tool),
            prompt_name="test",
        )
        context = _build_tool_context(session, resources)
        tool_call = _make_tool_call("mutating_fail", {"query": "hello"})

        result = runner.execute(tool_call, context=context)

        assert not result.success
        # State must be unchanged
        assert fs.read("file.txt").content == "original"
        assert session[SamplePlan].latest().objective == "test"

    def test_visibility_expansion_restores_state(self) -> None:
        """Acceptance criteria: VisibilityExpansionRequired restores state."""
        session = Session()
        fs = InMemoryFilesystem()
        fs.write("data.txt", "before")
        resources = ResourceRegistry.build({Filesystem: fs})
        state = ExecutionState(session=session, resources=resources)

        def expand_visibility(
            params: ToolParams, *, context: ToolContext
        ) -> ToolResult[ToolPayload]:
            if context.filesystem is not None:
                context.filesystem.write("data.txt", "during-expansion")
            raise _make_visibility_expansion()

        tool = Tool[ToolParams, ToolPayload](
            name="expanding",
            description="Expands visibility",
            handler=expand_visibility,
        )

        runner = ToolRunner(
            execution_state=state,
            tool_registry=_build_tool_registry(tool),
            prompt_name="test",
        )
        context = _build_tool_context(session, resources)
        tool_call = _make_tool_call("expanding", {"query": "hello"})

        with pytest.raises(VisibilityExpansionRequired):
            runner.execute(tool_call, context=context)

        # State should be restored
        assert fs.read("data.txt").content == "before"


class TestToolRunnerParamValidation:
    """Tests for parameter validation edge cases."""

    def test_tool_with_none_params_rejects_arguments(self) -> None:
        """Tool with params_type=None rejects any arguments."""

        def no_params_handler(
            params: None, *, context: ToolContext
        ) -> ToolResult[ToolPayload]:
            return ToolResult(message="ok", value=ToolPayload(answer="done"))

        tool = Tool[None, ToolPayload](
            name="no_params_tool",
            description="A tool that takes no params",
            handler=no_params_handler,
        )
        session = Session()
        state = ExecutionState(session=session)

        runner = ToolRunner(
            execution_state=state,
            tool_registry=_build_tool_registry(tool),
            prompt_name="test",
        )
        context = _build_tool_context(session)
        # Pass arguments to tool that doesn't accept them
        tool_call = _make_tool_call("no_params_tool", {"unexpected": "arg"})

        result = runner.execute(tool_call, context=context)

        assert not result.success
        assert "execution failed" in result.message

    def test_tool_with_none_params_succeeds_without_arguments(self) -> None:
        """Tool with params_type=None succeeds when no arguments provided."""

        def no_params_handler(
            params: None, *, context: ToolContext
        ) -> ToolResult[ToolPayload]:
            return ToolResult(message="ok", value=ToolPayload(answer="done"))

        tool = Tool[None, ToolPayload](
            name="no_params_tool",
            description="A tool that takes no params",
            handler=no_params_handler,
        )
        session = Session()
        state = ExecutionState(session=session)

        runner = ToolRunner(
            execution_state=state,
            tool_registry=_build_tool_registry(tool),
            prompt_name="test",
        )
        context = _build_tool_context(session)
        # Pass empty arguments
        tool_call = _make_tool_call("no_params_tool", {})

        result = runner.execute(tool_call, context=context)

        assert result.success
        assert result.value is not None
        assert result.value.answer == "done"

    def test_tool_param_parse_error(self) -> None:
        """Tool returns failure when param parsing fails."""
        tool = Tool[ToolParams, ToolPayload](
            name="test_tool",
            description="A test tool",
            handler=_success_handler,
        )
        session = Session()
        state = ExecutionState(session=session)

        runner = ToolRunner(
            execution_state=state,
            tool_registry=_build_tool_registry(tool),
            prompt_name="test",
        )
        context = _build_tool_context(session)
        # Pass extra forbidden field (extra="forbid" is used)
        tool_call = _make_tool_call(
            "test_tool", {"query": "valid", "forbidden_field": 1}
        )

        result = runner.execute(tool_call, context=context)

        assert not result.success
        assert "execution failed" in result.message


class TestToolRunnerDeadlineExpired:
    """Tests for deadline expiration before execution."""

    def test_deadline_expired_before_execution(self) -> None:
        """Tool fails when deadline has already expired."""
        from datetime import UTC, datetime, timedelta
        from unittest.mock import patch

        from weakincentives.adapters.core import PromptEvaluationError
        from weakincentives.budget import Deadline

        tool = Tool[ToolParams, ToolPayload](
            name="test_tool",
            description="A test tool",
            handler=_success_handler,
        )
        session = Session()
        state = ExecutionState(session=session)

        runner = ToolRunner(
            execution_state=state,
            tool_registry=_build_tool_registry(tool),
            prompt_name="test",
        )

        # Create a deadline that's 10 seconds in the future
        future_time = datetime.now(UTC) + timedelta(seconds=10)
        deadline = Deadline(expires_at=future_time)

        prompt_template = PromptTemplate[ToolPayload](
            ns="tests/tool-runner",
            key="tool-runner",
            name="test",
            sections=(
                MarkdownSection[ToolParams](
                    title="Task",
                    key="task",
                    template="Look up ${query}",
                    tools=[],
                ),
            ),
        )
        prompt = Prompt(prompt_template)
        context = ToolContext(
            prompt=prompt,
            rendered_prompt=None,
            adapter=cast(Any, object()),
            session=session,
            deadline=deadline,
            resources=ResourceRegistry(),
        )
        tool_call = _make_tool_call("test_tool", {"query": "hello"})

        # Mock _utcnow to return a time AFTER the deadline expires
        expired_time = future_time + timedelta(seconds=1)
        with patch("weakincentives.deadlines._utcnow", return_value=expired_time):
            with pytest.raises(PromptEvaluationError) as exc_info:
                runner.execute(tool_call, context=context)

        assert "Deadline expired" in str(exc_info.value)
