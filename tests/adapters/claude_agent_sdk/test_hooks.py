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

"""Tests for Claude Agent SDK hook implementations."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import pytest

from weakincentives.adapters.claude_agent_sdk._bridge import MCPToolExecutionState
from weakincentives.adapters.claude_agent_sdk._hooks import (
    HookConstraints,
    HookContext,
    create_post_tool_use_hook,
    create_pre_compact_hook,
    create_pre_tool_use_hook,
    create_stop_hook,
    create_subagent_stop_hook,
    create_task_completion_stop_hook,
    create_user_prompt_submit_hook,
    safe_hook_wrapper,
)
from weakincentives.adapters.claude_agent_sdk._task_completion import PlanBasedChecker
from weakincentives.budget import Budget, BudgetTracker
from weakincentives.contrib.tools.filesystem_memory import InMemoryFilesystem
from weakincentives.dataclasses import FrozenDataclass
from weakincentives.deadlines import Deadline
from weakincentives.filesystem import Filesystem
from weakincentives.prompt import (
    Feedback,
    FeedbackProviderConfig,
    FeedbackTrigger,
    Prompt,
    PromptTemplate,
)
from weakincentives.prompt.protocols import PromptProtocol
from weakincentives.runtime.events import InProcessDispatcher
from weakincentives.runtime.events.types import TokenUsage, ToolInvoked
from weakincentives.runtime.session import Session


# Mock Plan types for testing task completion checkers.
# These replicate the interface of the removed PlanningToolsSection types.
@FrozenDataclass()
class PlanStep:
    """Mock PlanStep for testing."""

    step_id: int
    title: str
    status: str = "pending"


@FrozenDataclass()
class Plan:
    """Mock Plan for testing."""

    objective: str
    status: str = "active"
    steps: tuple[PlanStep, ...] = ()


def _initialize_plan_session(session: Session) -> None:
    """Initialize session with Plan slice for testing."""
    session[Plan].seed(())


def _make_prompt() -> Prompt[object]:
    """Create a prompt in active context."""
    prompt: Prompt[object] = Prompt(PromptTemplate(ns="tests", key="hooks-test"))
    prompt.resources.__enter__()
    return prompt


def _make_prompt_with_fs(fs: InMemoryFilesystem) -> Prompt[object]:
    """Create a prompt with filesystem bound in active context."""
    prompt: Prompt[object] = Prompt(PromptTemplate(ns="tests", key="hooks-test"))
    prompt = prompt.bind(resources={Filesystem: fs})
    prompt.resources.__enter__()
    return prompt


class _AlwaysTriggerProvider:
    """Test provider that always triggers and returns fixed feedback."""

    @property
    def name(self) -> str:
        return "AlwaysTrigger"

    def should_run(
        self,
        *,
        context: Any,
    ) -> bool:
        return True

    def provide(
        self,
        *,
        context: Any,
    ) -> Feedback:
        return Feedback(
            provider_name=self.name,
            summary="Test feedback triggered",
            severity="info",
        )


def _make_prompt_with_feedback_provider() -> Prompt[object]:
    """Create a prompt with a feedback provider that always triggers."""
    provider = _AlwaysTriggerProvider()
    config = FeedbackProviderConfig(
        provider=provider,  # type: ignore[arg-type]
        trigger=FeedbackTrigger(every_n_calls=1),  # Trigger on every call
    )
    template: PromptTemplate[object] = PromptTemplate(
        ns="tests",
        key="hooks-test-feedback",
        name="test_prompt",  # Match the prompt_name used in HookContext
        feedback_providers=(config,),
    )
    prompt: Prompt[object] = Prompt(template)
    prompt.resources.__enter__()
    return prompt


@pytest.fixture
def session() -> Session:
    dispatcher = InProcessDispatcher()
    return Session(dispatcher=dispatcher)


@pytest.fixture
def hook_context(session: Session) -> HookContext:
    return HookContext(
        session=session,
        prompt=cast("PromptProtocol[object]", _make_prompt()),
        adapter_name="claude_agent_sdk",
        prompt_name="test_prompt",
    )


class TestHookContext:
    def test_basic_construction(self, session: Session) -> None:
        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )
        assert context.session is session
        assert context.adapter_name == "test_adapter"
        assert context.prompt_name == "test_prompt"
        assert context.deadline is None
        assert context.budget_tracker is None
        assert context.stop_reason is None
        assert context._tool_count == 0

    def test_with_deadline_and_budget(self, session: Session) -> None:
        from weakincentives.clock import FakeClock

        clock = FakeClock()
        anchor = datetime.now(UTC)
        clock.set_wall(anchor)
        deadline = Deadline(anchor + timedelta(minutes=5), clock=clock)
        budget = Budget(max_total_tokens=1000)
        tracker = BudgetTracker(budget)

        constraints = HookConstraints(deadline=deadline, budget_tracker=tracker)
        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
            constraints=constraints,
        )
        assert context.deadline is deadline
        assert context.budget_tracker is tracker

    def test_beat_with_heartbeat_configured(self, session: Session) -> None:
        from weakincentives.runtime.watchdog import Heartbeat

        beat_count = 0

        def on_beat() -> None:
            nonlocal beat_count
            beat_count += 1

        heartbeat = Heartbeat()
        heartbeat.add_callback(on_beat)

        constraints = HookConstraints(heartbeat=heartbeat)
        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
            constraints=constraints,
        )

        assert context.heartbeat is heartbeat
        context.beat()
        assert beat_count == 1

    def test_beat_without_heartbeat_is_noop(self, session: Session) -> None:
        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )

        assert context.heartbeat is None
        # Should not raise
        context.beat()


class TestPreToolUseHook:
    def test_allows_tool_by_default(self, hook_context: HookContext) -> None:
        hook = create_pre_tool_use_hook(hook_context)
        input_data = {
            "hook_event_name": "PreToolUse",
            "tool_name": "Read",
            "tool_input": {"path": "/test"},
        }

        result = asyncio.run(hook(input_data, "call-123", {"signal": None}))

        assert result == {}

    def test_denies_when_deadline_exceeded(self, session: Session) -> None:
        from weakincentives.clock import FakeClock

        clock = FakeClock()
        anchor = datetime.now(UTC)
        clock.set_wall(anchor)
        deadline = Deadline(anchor + timedelta(seconds=5), clock=clock)
        clock.advance(10)

        constraints = HookConstraints(deadline=deadline)
        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
            constraints=constraints,
        )
        hook = create_pre_tool_use_hook(context)
        input_data = {"hook_event_name": "PreToolUse", "tool_name": "Read"}

        result = asyncio.run(hook(input_data, "call-123", {"signal": None}))

        assert result.get("hookSpecificOutput", {}).get("permissionDecision") == "deny"
        assert "Deadline exceeded" in result.get("hookSpecificOutput", {}).get(
            "permissionDecisionReason", ""
        )

    def test_denies_when_budget_exhausted(self, session: Session) -> None:
        budget = Budget(max_total_tokens=100)
        tracker = BudgetTracker(budget)
        tracker.record_cumulative(
            "eval1", TokenUsage(input_tokens=100, output_tokens=50)
        )

        constraints = HookConstraints(budget_tracker=tracker)
        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
            constraints=constraints,
        )
        hook = create_pre_tool_use_hook(context)
        input_data = {"hook_event_name": "PreToolUse", "tool_name": "Read"}

        result = asyncio.run(hook(input_data, "call-123", {"signal": None}))

        assert result.get("hookSpecificOutput", {}).get("permissionDecision") == "deny"
        assert "budget exhausted" in result.get("hookSpecificOutput", {}).get(
            "permissionDecisionReason", ""
        )

    def test_allows_with_remaining_budget(self, session: Session) -> None:
        budget = Budget(max_total_tokens=1000)
        tracker = BudgetTracker(budget)
        tracker.record_cumulative(
            "eval1", TokenUsage(input_tokens=50, output_tokens=50)
        )

        constraints = HookConstraints(budget_tracker=tracker)
        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
            constraints=constraints,
        )
        hook = create_pre_tool_use_hook(context)
        input_data = {"hook_event_name": "PreToolUse", "tool_name": "Read"}

        result = asyncio.run(hook(input_data, "call-123", {"signal": None}))

        assert result == {}


class TestPostToolUseHook:
    def test_publishes_tool_invoked_event(self, session: Session) -> None:
        events: list[ToolInvoked] = []
        session.dispatcher.subscribe(ToolInvoked, lambda e: events.append(e))

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )
        hook = create_post_tool_use_hook(context)
        input_data = {
            "hook_event_name": "PostToolUse",
            "tool_name": "Read",
            "tool_input": {"path": "/test.txt"},
            "tool_response": {"stdout": "file contents"},
        }

        asyncio.run(hook(input_data, "call-123", {"signal": None}))

        assert len(events) == 1
        event = events[0]
        assert event.name == "Read"
        assert event.params == {"path": "/test.txt"}
        assert event.result == {"stdout": "file contents"}
        assert event.call_id == "call-123"
        assert event.adapter == "test_adapter"
        assert event.prompt_name == "test_prompt"

    def test_tracks_tool_count(self, hook_context: HookContext) -> None:
        hook = create_post_tool_use_hook(hook_context)

        assert hook_context._tool_count == 0

        asyncio.run(
            hook(
                {
                    "hook_event_name": "PostToolUse",
                    "tool_name": "Read",
                    "tool_input": {},
                    "tool_response": {},
                },
                None,
                {"signal": None},
            )
        )
        assert hook_context._tool_count == 1

        asyncio.run(
            hook(
                {
                    "hook_event_name": "PostToolUse",
                    "tool_name": "Write",
                    "tool_input": {},
                    "tool_response": {},
                },
                None,
                {"signal": None},
            )
        )
        assert hook_context._tool_count == 2

    def test_handles_tool_error(self, session: Session) -> None:
        events: list[ToolInvoked] = []
        session.dispatcher.subscribe(ToolInvoked, lambda e: events.append(e))

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )
        hook = create_post_tool_use_hook(context)
        input_data = {
            "hook_event_name": "PostToolUse",
            "tool_name": "Read",
            "tool_input": {"path": "/missing.txt"},
            "tool_response": {"stderr": "File not found"},
        }

        asyncio.run(hook(input_data, "call-456", {"signal": None}))

        assert len(events) == 1
        event = events[0]
        assert event.name == "Read"
        assert event.call_id == "call-456"

    def test_handles_non_dict_tool_response(self, session: Session) -> None:
        events: list[ToolInvoked] = []
        session.dispatcher.subscribe(ToolInvoked, lambda e: events.append(e))

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )
        hook = create_post_tool_use_hook(context)
        # tool_response is a non-dict, non-None value (e.g., a string)
        input_data = {
            "hook_event_name": "PostToolUse",
            "tool_name": "Echo",
            "tool_input": {"message": "hello"},
            "tool_response": "hello world",  # Non-dict response
        }

        asyncio.run(hook(input_data, "call-789", {"signal": None}))

        assert len(events) == 1
        event = events[0]
        assert event.name == "Echo"
        assert event.rendered_output == "hello world"

    def test_truncates_long_output(self, session: Session) -> None:
        events: list[ToolInvoked] = []
        session.dispatcher.subscribe(ToolInvoked, lambda e: events.append(e))

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )
        hook = create_post_tool_use_hook(context)
        long_output = "x" * 2000
        input_data = {
            "hook_event_name": "PostToolUse",
            "tool_name": "Read",
            "tool_input": {},
            "tool_response": {"stdout": long_output},
        }

        asyncio.run(hook(input_data, None, {"signal": None}))

        assert len(events) == 1
        assert len(events[0].rendered_output) == 1000

    def test_stops_on_structured_output_by_default(self, session: Session) -> None:
        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )
        hook = create_post_tool_use_hook(context)
        input_data = {
            "hook_event_name": "PostToolUse",
            "tool_name": "StructuredOutput",
            "tool_input": {"output": {"key": "value"}},
            "tool_response": {"stdout": ""},
        }

        result = asyncio.run(hook(input_data, "call-structured", {"signal": None}))

        # SDK uses continue_ to avoid Python keyword conflict
        assert result == {"continue_": False}

    def test_does_not_stop_on_structured_output_when_disabled(
        self, session: Session
    ) -> None:
        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )
        hook = create_post_tool_use_hook(context, stop_on_structured_output=False)
        input_data = {
            "hook_event_name": "PostToolUse",
            "tool_name": "StructuredOutput",
            "tool_input": {"output": {"key": "value"}},
            "tool_response": {"stdout": ""},
        }

        result = asyncio.run(hook(input_data, "call-structured", {"signal": None}))

        assert result == {}

    def test_skips_mcp_wink_tools(self, session: Session) -> None:
        """MCP-bridged WINK tools should not publish ToolInvoked events (they do it themselves)."""
        events: list[ToolInvoked] = []
        session.dispatcher.subscribe(ToolInvoked, lambda e: events.append(e))

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )
        hook = create_post_tool_use_hook(context)
        input_data = {
            "hook_event_name": "PostToolUse",
            "tool_name": "mcp__wink__planning_setup_plan",
            "tool_input": {"objective": "test plan"},
            "tool_response": {"stdout": "Plan created"},
        }

        result = asyncio.run(hook(input_data, "call-mcp", {"signal": None}))

        # Should return empty without publishing ToolInvoked event
        assert result == {}
        assert len(events) == 0
        # Tool count IS incremented (needed for feedback provider triggers)
        assert context._tool_count == 1

    def test_skips_mcp_wink_tools_with_parsed_input(self, session: Session) -> None:
        """MCP-bridged WINK tools should be skipped even with full SDK input format."""
        events: list[ToolInvoked] = []
        session.dispatcher.subscribe(ToolInvoked, lambda e: events.append(e))

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )
        hook = create_post_tool_use_hook(context)
        # Full SDK-format input that will be parsed successfully
        input_data = {
            "hook_event_name": "PostToolUse",
            "session_id": "sess-123",
            "tool_name": "mcp__wink__open_sections",
            "tool_input": {"section_keys": ["reference-docs"]},
            "tool_response": {"stdout": "Sections opened"},
            "cwd": "/home",
        }

        result = asyncio.run(hook(input_data, "call-mcp-full", {"signal": None}))

        assert result == {}
        assert len(events) == 0

    def test_handles_mcp_tool_post_without_error(self, session: Session) -> None:
        """PostToolUse handles MCP tools without error."""
        mcp_state = MCPToolExecutionState()
        # Note: With queue-based approach, PreToolUse enqueues and BridgedTool dequeues.
        # PostToolUse just runs feedback providers - no state management needed.

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
            constraints=HookConstraints(mcp_tool_state=mcp_state),
        )
        hook = create_post_tool_use_hook(context)
        input_data = {
            "hook_event_name": "PostToolUse",
            "tool_name": "mcp__wink__planning_update",
            "tool_input": {"step_id": 1},
            "tool_response": {"stdout": "Updated"},
        }

        # Should complete without error
        result = asyncio.run(hook(input_data, "call-mcp-456", {"signal": None}))
        assert result == {}

    def test_returns_context_when_structured_output_with_incomplete_tasks(
        self, session: Session
    ) -> None:
        """PostToolUse returns additionalContext for StructuredOutput when tasks incomplete."""
        _initialize_plan_session(session)
        session.dispatch(
            Plan(
                objective="Test objective",
                status="active",
                steps=(
                    PlanStep(step_id=1, title="Done task", status="done"),
                    PlanStep(step_id=2, title="Pending task", status="pending"),
                ),
            )
        )

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )
        hook = create_post_tool_use_hook(
            context, task_completion_checker=PlanBasedChecker(plan_type=Plan)
        )
        input_data = {
            "hook_event_name": "PostToolUse",
            "tool_name": "StructuredOutput",
            "tool_input": {"output": {"key": "value"}},
            "tool_response": {"stdout": ""},
        }

        result = asyncio.run(hook(input_data, "call-structured", {"signal": None}))

        # Should return continue_: True to force continuation (SDK converts to "continue")
        assert result.get("continue_") is True
        # Should return additionalContext with feedback
        hook_output = result.get("hookSpecificOutput", {})
        assert hook_output.get("hookEventName") == "PostToolUse"
        additional_context = hook_output.get("additionalContext", "")
        assert "incomplete" in additional_context.lower()
        assert "Pending task" in additional_context

    def test_stops_when_structured_output_with_complete_tasks(
        self, session: Session
    ) -> None:
        """PostToolUse stops after StructuredOutput when all tasks complete."""
        _initialize_plan_session(session)
        session.dispatch(
            Plan(
                objective="Test objective",
                status="completed",
                steps=(
                    PlanStep(step_id=1, title="Task 1", status="done"),
                    PlanStep(step_id=2, title="Task 2", status="done"),
                ),
            )
        )

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )
        hook = create_post_tool_use_hook(
            context, task_completion_checker=PlanBasedChecker(plan_type=Plan)
        )
        input_data = {
            "hook_event_name": "PostToolUse",
            "tool_name": "StructuredOutput",
            "tool_input": {"output": {"key": "value"}},
            "tool_response": {"stdout": ""},
        }

        result = asyncio.run(hook(input_data, "call-structured", {"signal": None}))

        # Should stop - all tasks complete
        assert result == {"continue_": False}

    def test_stops_when_structured_output_without_plan(self, session: Session) -> None:
        """PostToolUse stops after StructuredOutput when no plan exists."""
        # Don't initialize Plan slice

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )
        hook = create_post_tool_use_hook(
            context, task_completion_checker=PlanBasedChecker(plan_type=Plan)
        )
        input_data = {
            "hook_event_name": "PostToolUse",
            "tool_name": "StructuredOutput",
            "tool_input": {"output": {"key": "value"}},
            "tool_response": {"stdout": ""},
        }

        result = asyncio.run(hook(input_data, "call-structured", {"signal": None}))

        # Should stop - no plan means nothing to enforce
        assert result == {"continue_": False}

    def test_returns_feedback_when_provider_triggers(self, session: Session) -> None:
        """PostToolUse returns additionalContext when feedback provider triggers."""
        # Initialize ToolInvoked slice with empty tuple for tool_call_count
        session[ToolInvoked].seed(())

        prompt = _make_prompt_with_feedback_provider()
        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", prompt),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )
        hook = create_post_tool_use_hook(context)

        # Use a regular tool (not StructuredOutput) so we hit the feedback path
        input_data = {
            "hook_event_name": "PostToolUse",
            "tool_name": "Read",
            "tool_input": {"file_path": "/test.txt"},
            "tool_response": {"stdout": "file content"},
        }

        result = asyncio.run(hook(input_data, "call-read", {"signal": None}))

        # Should return additionalContext with feedback
        hook_output = result.get("hookSpecificOutput", {})
        assert hook_output.get("hookEventName") == "PostToolUse"
        additional_context = hook_output.get("additionalContext", "")
        assert "Test feedback triggered" in additional_context

    def test_returns_feedback_for_mcp_tool_when_provider_triggers(
        self, session: Session
    ) -> None:
        """PostToolUse returns additionalContext for MCP tools when feedback triggers."""
        # Seed ToolInvoked with an existing event so feedback provider triggers.
        # MCP tools dispatch their ToolInvoked via the bridge, so we simulate that
        # by pre-seeding an event. The feedback trigger checks tool_call_count.
        existing_event = ToolInvoked(
            prompt_name="test_prompt",
            adapter="claude_agent_sdk",
            name="mcp__wink__some_tool",
            params={},
            result={},
            session_id=None,
            created_at=datetime.now(UTC),
            call_id="prev-call",
        )
        session[ToolInvoked].seed((existing_event,))

        prompt = _make_prompt_with_feedback_provider()
        mcp_tool_state = MCPToolExecutionState()
        constraints = HookConstraints(mcp_tool_state=mcp_tool_state)
        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", prompt),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
            constraints=constraints,
        )
        hook = create_post_tool_use_hook(context)

        # Use an MCP tool name to hit the _handle_mcp_tool_post path
        input_data = {
            "hook_event_name": "PostToolUse",
            "tool_name": "mcp__wink__planning_setup_plan",
            "tool_input": {"plan": "test plan"},
            "tool_response": {"success": True},
        }

        result = asyncio.run(hook(input_data, "call-mcp", {"signal": None}))

        # Should return additionalContext with feedback
        hook_output = result.get("hookSpecificOutput", {})
        assert hook_output.get("hookEventName") == "PostToolUse"
        additional_context = hook_output.get("additionalContext", "")
        assert "Test feedback triggered" in additional_context


class TestUserPromptSubmitHook:
    def test_returns_empty_by_default(self, hook_context: HookContext) -> None:
        hook = create_user_prompt_submit_hook(hook_context)
        input_data = {"hook_event_name": "UserPromptSubmit", "prompt": "Do something"}

        result = asyncio.run(hook(input_data, None, {"signal": None}))

        assert result == {}


class TestStopHook:
    def test_records_stop_reason(self, hook_context: HookContext) -> None:
        hook = create_stop_hook(hook_context)
        # SDK StopHookInput doesn't have stopReason field, so hook defaults to end_turn
        input_data = {"hook_event_name": "Stop", "stop_hook_active": False}

        assert hook_context.stop_reason is None

        asyncio.run(hook(input_data, None, {"signal": None}))

        # StopHookInput doesn't have stopReason; hook always sets end_turn
        assert hook_context.stop_reason == "end_turn"

    def test_defaults_to_end_turn(self, hook_context: HookContext) -> None:
        hook = create_stop_hook(hook_context)
        input_data = {"hook_event_name": "Stop"}

        asyncio.run(hook(input_data, None, {"signal": None}))

        assert hook_context.stop_reason == "end_turn"


class TestSafeHookWrapper:
    def test_passes_through_successful_result(self, hook_context: HookContext) -> None:
        async def success_hook(
            input_data: dict[str, Any],
            tool_use_id: str | None,
            context: Any,
        ) -> dict[str, Any]:
            return {"result": "success"}

        # safe_hook_wrapper uses asyncio.get_event_loop() which requires an active loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = safe_hook_wrapper(
                success_hook,
                {"hook_event_name": "PreToolUse", "tool_name": "test"},
                "call-123",
                hook_context,
            )
        finally:
            loop.close()

        assert result == {"result": "success"}

    def test_catches_deadline_exceeded(self, hook_context: HookContext) -> None:
        class DeadlineExceededError(Exception):
            pass

        async def deadline_hook(
            input_data: dict[str, Any],
            tool_use_id: str | None,
            context: Any,
        ) -> dict[str, Any]:
            raise DeadlineExceededError("Deadline exceeded")

        # safe_hook_wrapper uses asyncio.get_event_loop() which requires an active loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = safe_hook_wrapper(
                deadline_hook,
                {"hook_event_name": "PreToolUse", "tool_name": "test"},
                "call-123",
                hook_context,
            )
        finally:
            loop.close()

        assert result.get("hookSpecificOutput", {}).get("permissionDecision") == "deny"
        assert "Deadline exceeded" in result.get("hookSpecificOutput", {}).get(
            "permissionDecisionReason", ""
        )

    def test_catches_budget_exhausted(self, hook_context: HookContext) -> None:
        class BudgetExhaustedError(Exception):
            pass

        async def budget_hook(
            input_data: dict[str, Any],
            tool_use_id: str | None,
            context: Any,
        ) -> dict[str, Any]:
            raise BudgetExhaustedError("Budget exhausted")

        # safe_hook_wrapper uses asyncio.get_event_loop() which requires an active loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = safe_hook_wrapper(
                budget_hook,
                {"hook_event_name": "PreToolUse", "tool_name": "test"},
                "call-123",
                hook_context,
            )
        finally:
            loop.close()

        assert result.get("hookSpecificOutput", {}).get("permissionDecision") == "deny"
        assert "Budget exhausted" in result.get("hookSpecificOutput", {}).get(
            "permissionDecisionReason", ""
        )

    def test_catches_unknown_errors(self, hook_context: HookContext) -> None:
        async def failing_hook(
            input_data: dict[str, Any],
            tool_use_id: str | None,
            context: Any,
        ) -> dict[str, Any]:
            raise RuntimeError("Unexpected error")

        # safe_hook_wrapper uses asyncio.get_event_loop() which requires an active loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = safe_hook_wrapper(
                failing_hook,
                {"hook_event_name": "PreToolUse", "tool_name": "test"},
                "call-123",
                hook_context,
            )
        finally:
            loop.close()

        assert result == {}


class TestPostToolUseHookWithTypedParsing:
    def test_publishes_event_with_raw_result(self, session: Session) -> None:
        """Test that hook stores raw dict as result, not typed dataclass."""
        events: list[ToolInvoked] = []
        session.dispatcher.subscribe(ToolInvoked, lambda e: events.append(e))

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )
        hook = create_post_tool_use_hook(context)
        # Full SDK-format input
        input_data = {
            "hook_event_name": "PostToolUse",
            "session_id": "sess-123",
            "tool_name": "Read",
            "tool_input": {"path": "/test.txt"},
            "tool_response": {
                "stdout": "file contents",
                "stderr": "",
                "interrupted": False,
                "isImage": False,
            },
            "cwd": "/home",
            "transcript_path": "/transcript",
        }

        asyncio.run(hook(input_data, "call-typed", {"signal": None}))

        assert len(events) == 1
        event = events[0]
        assert event.name == "Read"
        # result should be the raw dict (SDK native tools)
        assert event.result == {
            "stdout": "file contents",
            "stderr": "",
            "interrupted": False,
            "isImage": False,
        }

    def test_fallback_when_missing_tool_name(self, session: Session) -> None:
        """Test fallback to dict access when parsing fails (missing required field)."""
        events: list[ToolInvoked] = []
        session.dispatcher.subscribe(ToolInvoked, lambda e: events.append(e))

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )
        hook = create_post_tool_use_hook(context)
        # Malformed input missing tool_name
        input_data = {
            "hook_event_name": "PostToolUse",
            "tool_input": {"key": "value"},
            "tool_response": {"stdout": "output"},
        }

        asyncio.run(hook(input_data, "call-fallback", {"signal": None}))

        assert len(events) == 1
        event = events[0]
        # Falls back to dict access, tool_name defaults to ""
        assert event.name == ""

    def test_fallback_with_string_tool_response(self, session: Session) -> None:
        """Test fallback when parsing fails and tool_response is a string."""
        events: list[ToolInvoked] = []
        session.dispatcher.subscribe(ToolInvoked, lambda e: events.append(e))

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )
        hook = create_post_tool_use_hook(context)
        # Missing tool_name causes parsing failure, and tool_response is a string
        input_data = {
            "hook_event_name": "PostToolUse",
            "tool_input": {"key": "value"},
            "tool_response": "string output",  # Non-dict
        }

        asyncio.run(hook(input_data, "call-string", {"signal": None}))

        assert len(events) == 1
        event = events[0]
        assert event.name == ""
        assert event.rendered_output == "string output"

    def test_fallback_with_none_tool_response(self, session: Session) -> None:
        """Test fallback when parsing fails and tool_response is None."""
        events: list[ToolInvoked] = []
        session.dispatcher.subscribe(ToolInvoked, lambda e: events.append(e))

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )
        hook = create_post_tool_use_hook(context)
        # Missing tool_name causes parsing failure, and tool_response is None
        input_data = {
            "hook_event_name": "PostToolUse",
            "tool_input": {"key": "value"},
            "tool_response": None,
        }

        asyncio.run(hook(input_data, "call-none", {"signal": None}))

        assert len(events) == 1
        event = events[0]
        assert event.name == ""
        assert event.rendered_output == ""


class TestSubagentStopHook:
    def test_returns_empty_dict(self, hook_context: HookContext) -> None:
        hook = create_subagent_stop_hook(hook_context)
        result = asyncio.run(
            hook(
                {"hook_event_name": "SubagentStop", "session_id": "test"},
                None,
                {"signal": None},
            )
        )

        assert result == {}


class TestPreCompactHook:
    def test_returns_empty_dict(self, hook_context: HookContext) -> None:
        hook = create_pre_compact_hook(hook_context)
        result = asyncio.run(
            hook(
                {"hook_event_name": "PreCompact", "session_id": "test"},
                None,
                {"signal": None},
            )
        )

        assert result == {}

    def test_increments_compact_count(self, hook_context: HookContext) -> None:
        assert hook_context.stats.compact_count == 0
        hook = create_pre_compact_hook(hook_context)

        asyncio.run(
            hook(
                {"hook_event_name": "PreCompact", "session_id": "test"},
                None,
                {"signal": None},
            )
        )
        assert hook_context.stats.compact_count == 1

        asyncio.run(
            hook(
                {"hook_event_name": "PreCompact", "session_id": "test2"},
                None,
                {"signal": None},
            )
        )
        assert hook_context.stats.compact_count == 2


class TestPreToolUseHookTransactional:
    """Tests for pre-tool hook transactional snapshot functionality."""

    def test_takes_snapshot_with_session_and_prompt(self, session: Session) -> None:
        """Pre-tool hook takes snapshot when session and prompt are present."""
        fs = InMemoryFilesystem()
        fs.write("/test.txt", "initial")
        prompt = _make_prompt_with_fs(fs)

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", prompt),
            adapter_name="claude_agent_sdk",
            prompt_name="test_prompt",
        )

        hook = create_pre_tool_use_hook(context)
        result = asyncio.run(
            hook(
                {"hook_event_name": "PreToolUse", "tool_name": "Edit"},
                "tool-123",
                {"signal": None},
            )
        )

        assert result == {}
        assert "tool-123" in context._tracker._pending_tools
        pending = context._tracker._pending_tools["tool-123"]
        assert pending.tool_name == "Edit"

    def test_skips_snapshot_for_mcp_wink_tools(self, session: Session) -> None:
        """Pre-tool hook skips snapshot for MCP WINK tools."""
        fs = InMemoryFilesystem()
        prompt = _make_prompt_with_fs(fs)

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", prompt),
            adapter_name="claude_agent_sdk",
            prompt_name="test_prompt",
        )

        hook = create_pre_tool_use_hook(context)
        asyncio.run(
            hook(
                {"hook_event_name": "PreToolUse", "tool_name": "mcp__wink__search"},
                "tool-123",
                {"signal": None},
            )
        )

        # Should NOT have pending execution for MCP tools (tracker not even initialized)
        assert (
            context._tool_tracker is None
            or "tool-123" not in context._tracker._pending_tools
        )

    def test_enqueues_mcp_tool_state_for_mcp_tools(self, session: Session) -> None:
        """Pre-tool hook enqueues tool_use_id on mcp_tool_state for MCP tools."""
        mcp_state = MCPToolExecutionState()
        tool_input = {"objective": "test plan"}

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="claude_agent_sdk",
            prompt_name="test_prompt",
            constraints=HookConstraints(mcp_tool_state=mcp_state),
        )

        hook = create_pre_tool_use_hook(context)
        asyncio.run(
            hook(
                {
                    "hook_event_name": "PreToolUse",
                    "tool_name": "mcp__wink__planning",
                    "tool_input": tool_input,
                },
                "call-mcp-123",
                {"signal": None},
            )
        )

        # mcp_tool_state should have the tool_use_id enqueued for this tool+params
        assert mcp_state.dequeue("planning", tool_input) == "call-mcp-123"

    def test_does_not_enqueue_mcp_tool_state_for_native_tools(
        self, session: Session
    ) -> None:
        """Pre-tool hook does not enqueue tool_use_id for native SDK tools."""
        mcp_state = MCPToolExecutionState()
        tool_input = {"file_path": "/test.txt"}

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="claude_agent_sdk",
            prompt_name="test_prompt",
            constraints=HookConstraints(mcp_tool_state=mcp_state),
        )

        hook = create_pre_tool_use_hook(context)
        asyncio.run(
            hook(
                {
                    "hook_event_name": "PreToolUse",
                    "tool_name": "Read",
                    "tool_input": tool_input,
                },
                "call-native-123",
                {"signal": None},
            )
        )

        # mcp_tool_state should remain empty for native tools
        assert mcp_state.dequeue("Read", tool_input) is None


class TestPostToolUseHookTransactional:
    """Tests for post-tool hook transactional restore functionality."""

    def test_restores_state_on_tool_failure(self, session: Session) -> None:
        """Post-tool hook restores state when tool fails."""
        fs = InMemoryFilesystem()
        fs.write("/test.txt", "initial")
        prompt = _make_prompt_with_fs(fs)

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", prompt),
            adapter_name="claude_agent_sdk",
            prompt_name="test_prompt",
        )

        # Take snapshot via pre-tool hook
        pre_hook = create_pre_tool_use_hook(context)
        asyncio.run(
            pre_hook(
                {"hook_event_name": "PreToolUse", "tool_name": "Edit"},
                "tool-123",
                {"signal": None},
            )
        )

        # Modify state (simulates tool making changes)
        fs.write("/test.txt", "modified")

        # Post-tool hook with failure (stderr indicates error)
        post_hook = create_post_tool_use_hook(context)
        asyncio.run(
            post_hook(
                {
                    "hook_event_name": "PostToolUse",
                    "tool_name": "Edit",
                    "tool_input": {},
                    "tool_response": {"stderr": "Error: file not found"},
                },
                "tool-123",
                {"signal": None},
            )
        )

        # State should be restored
        assert fs.read("/test.txt").content == "initial"
        assert "tool-123" not in context._tracker._pending_tools

    def test_no_restore_on_success(self, session: Session) -> None:
        """Post-tool hook doesn't restore state on success."""
        fs = InMemoryFilesystem()
        fs.write("/test.txt", "initial")
        prompt = _make_prompt_with_fs(fs)

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", prompt),
            adapter_name="claude_agent_sdk",
            prompt_name="test_prompt",
        )

        # Take snapshot via pre-tool hook
        pre_hook = create_pre_tool_use_hook(context)
        asyncio.run(
            pre_hook(
                {"hook_event_name": "PreToolUse", "tool_name": "Edit"},
                "tool-123",
                {"signal": None},
            )
        )

        # Modify state (simulates tool making changes)
        fs.write("/test.txt", "modified")

        # Post-tool hook with success
        post_hook = create_post_tool_use_hook(context)
        asyncio.run(
            post_hook(
                {
                    "hook_event_name": "PostToolUse",
                    "tool_name": "Edit",
                    "tool_input": {},
                    "tool_response": {"stdout": "File edited successfully"},
                },
                "tool-123",
                {"signal": None},
            )
        )

        # State should NOT be restored
        assert fs.read("/test.txt").content == "modified"
        assert "tool-123" not in context._tracker._pending_tools


class TestIsToolErrorResponse:
    """Tests for _is_tool_error_response helper function."""

    def test_non_dict_returns_false(self) -> None:
        """Non-dict responses are not considered errors."""
        from weakincentives.adapters.claude_agent_sdk._hooks import (
            _is_tool_error_response,
        )

        assert _is_tool_error_response("string") is False
        assert _is_tool_error_response(123) is False
        assert _is_tool_error_response(None) is False

    def test_is_error_flag(self) -> None:
        """is_error flag indicates error."""
        from weakincentives.adapters.claude_agent_sdk._hooks import (
            _is_tool_error_response,
        )

        assert _is_tool_error_response({"is_error": True}) is True
        assert _is_tool_error_response({"is_error": False}) is False

    def test_isError_flag(self) -> None:
        """isError flag indicates error."""
        from weakincentives.adapters.claude_agent_sdk._hooks import (
            _is_tool_error_response,
        )

        assert _is_tool_error_response({"isError": True}) is True
        assert _is_tool_error_response({"isError": False}) is False

    def test_error_in_content(self) -> None:
        """Error text in content indicates error."""
        from weakincentives.adapters.claude_agent_sdk._hooks import (
            _is_tool_error_response,
        )

        assert (
            _is_tool_error_response(
                {"content": [{"type": "text", "text": "Error: something went wrong"}]}
            )
            is True
        )

        assert (
            _is_tool_error_response(
                {"content": [{"type": "text", "text": "error - file not found"}]}
            )
            is True
        )

        # Normal content is not an error
        assert (
            _is_tool_error_response(
                {"content": [{"type": "text", "text": "File created successfully"}]}
            )
            is False
        )

    def test_empty_content(self) -> None:
        """Empty content is not an error."""
        from weakincentives.adapters.claude_agent_sdk._hooks import (
            _is_tool_error_response,
        )

        assert _is_tool_error_response({"content": []}) is False
        assert _is_tool_error_response({}) is False

    def test_content_with_non_dict_item(self) -> None:
        """Content with non-dict first item is not considered an error."""
        from weakincentives.adapters.claude_agent_sdk._hooks import (
            _is_tool_error_response,
        )

        # First item is a string instead of dict
        assert _is_tool_error_response({"content": ["not a dict"]}) is False
        # First item is a number instead of dict
        assert _is_tool_error_response({"content": [123]}) is False
        # First item is None instead of dict
        assert _is_tool_error_response({"content": [None]}) is False

    def test_content_with_non_string_text(self) -> None:
        """Content dict with non-string text field is not considered an error."""
        from weakincentives.adapters.claude_agent_sdk._hooks import (
            _is_tool_error_response,
        )

        # Text field is a number instead of string
        assert _is_tool_error_response({"content": [{"text": 123}]}) is False
        # Text field is None instead of string
        assert _is_tool_error_response({"content": [{"text": None}]}) is False
        # Text field is a list instead of string
        assert _is_tool_error_response({"content": [{"text": ["error"]}]}) is False


class TestTaskCompletionStopHook:
    """Tests for task completion stop hook functionality."""

    def test_allows_stop_when_no_plan_slice(self, hook_context: HookContext) -> None:
        """Stop is allowed when Plan slice is not installed in session."""
        hook = create_task_completion_stop_hook(
            hook_context, checker=PlanBasedChecker(plan_type=Plan)
        )
        input_data = {"hook_event_name": "Stop", "stopReason": "end_turn"}

        result = asyncio.run(hook(input_data, None, {"signal": None}))

        assert result == {}
        assert hook_context.stop_reason == "end_turn"

    def test_allows_stop_when_no_plan_initialized(self, session: Session) -> None:
        """Stop is allowed when Plan slice exists but no plan is initialized."""
        # Install Plan slice but don't initialize a plan
        _initialize_plan_session(session)

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )

        hook = create_task_completion_stop_hook(
            context, checker=PlanBasedChecker(plan_type=Plan)
        )
        input_data = {"hook_event_name": "Stop", "stopReason": "end_turn"}

        result = asyncio.run(hook(input_data, None, {"signal": None}))

        assert result == {}
        assert context.stop_reason == "end_turn"

    def test_allows_stop_when_all_tasks_complete(self, session: Session) -> None:
        """Stop is allowed when all plan tasks are marked as done."""
        _initialize_plan_session(session)

        # Create a plan with all steps done
        completed_plan = Plan(
            objective="Test objective",
            status="completed",
            steps=(
                PlanStep(step_id=1, title="Task 1", status="done"),
                PlanStep(step_id=2, title="Task 2", status="done"),
            ),
        )
        session.dispatch(completed_plan)

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )

        hook = create_task_completion_stop_hook(
            context, checker=PlanBasedChecker(plan_type=Plan)
        )
        input_data = {"hook_event_name": "Stop", "stopReason": "end_turn"}

        result = asyncio.run(hook(input_data, None, {"signal": None}))

        assert result == {}
        assert context.stop_reason == "end_turn"

    def test_signals_continue_when_tasks_incomplete(self, session: Session) -> None:
        """Stop hook signals continuation when tasks are incomplete."""
        _initialize_plan_session(session)

        # Create a plan with incomplete steps
        incomplete_plan = Plan(
            objective="Test objective",
            status="active",
            steps=(
                PlanStep(step_id=1, title="Done task", status="done"),
                PlanStep(step_id=2, title="Pending task", status="pending"),
                PlanStep(step_id=3, title="In progress task", status="in_progress"),
            ),
        )
        session.dispatch(incomplete_plan)

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )

        hook = create_task_completion_stop_hook(
            context, checker=PlanBasedChecker(plan_type=Plan)
        )
        input_data = {"hook_event_name": "Stop", "stopReason": "end_turn"}

        result = asyncio.run(hook(input_data, None, {"signal": None}))

        # Should return continue_: True to force continuation (SDK converts to "continue")
        assert result.get("continue_") is True
        assert "2 incomplete task(s)" in result.get("reason", "")
        assert "Pending task" in result.get("reason", "")
        assert context.stop_reason == "end_turn"

    def test_records_stop_reason(self, session: Session) -> None:
        """Stop hook records the stop reason regardless of task state."""
        _initialize_plan_session(session)

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )

        hook = create_task_completion_stop_hook(
            context, checker=PlanBasedChecker(plan_type=Plan)
        )
        # SDK StopHookInput doesn't have stopReason field; hook defaults to end_turn
        input_data = {"hook_event_name": "Stop", "stop_hook_active": False}

        assert context.stop_reason is None
        asyncio.run(hook(input_data, None, {"signal": None}))
        # StopHookInput doesn't have stopReason; hook always sets end_turn
        assert context.stop_reason == "end_turn"

    def test_defaults_stop_reason_to_end_turn(self, session: Session) -> None:
        """Stop hook defaults stopReason to end_turn when not provided."""
        _initialize_plan_session(session)

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )

        hook = create_task_completion_stop_hook(
            context, checker=PlanBasedChecker(plan_type=Plan)
        )

        asyncio.run(hook({"hook_event_name": "Stop"}, None, {"signal": None}))

        assert context.stop_reason == "end_turn"

    def test_handles_empty_steps(self, session: Session) -> None:
        """Stop is allowed when plan has no steps."""
        _initialize_plan_session(session)

        # Create a plan with no steps
        empty_plan = Plan(
            objective="Test objective",
            status="active",
            steps=(),
        )
        session.dispatch(empty_plan)

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )

        hook = create_task_completion_stop_hook(
            context, checker=PlanBasedChecker(plan_type=Plan)
        )
        input_data = {"hook_event_name": "Stop", "stopReason": "end_turn"}

        result = asyncio.run(hook(input_data, None, {"signal": None}))

        assert result == {}

    def test_reminder_message_truncates_long_task_list(self, session: Session) -> None:
        """Reminder message truncates task titles when there are many incomplete."""
        _initialize_plan_session(session)

        # Create a plan with many incomplete steps
        many_steps_plan = Plan(
            objective="Test objective",
            status="active",
            steps=tuple(
                PlanStep(step_id=i, title=f"Task {i}", status="pending")
                for i in range(1, 10)
            ),
        )
        session.dispatch(many_steps_plan)

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )

        hook = create_task_completion_stop_hook(
            context, checker=PlanBasedChecker(plan_type=Plan)
        )
        input_data = {"hook_event_name": "Stop", "stopReason": "end_turn"}

        result = asyncio.run(hook(input_data, None, {"signal": None}))

        # Should return continue_: True to force continuation (SDK converts to "continue")
        assert result.get("continue_") is True
        reason = result.get("reason", "")
        # Should contain first 3 tasks and ellipsis
        assert "Task 1" in reason
        assert "Task 2" in reason
        assert "Task 3" in reason
        assert "..." in reason
        # Should not list all tasks
        assert "Task 9" not in reason

    def test_checker_protocol_with_plan_based_checker(self, session: Session) -> None:
        """Hook accepts any TaskCompletionChecker implementation."""
        _initialize_plan_session(session)

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )

        # Create hook with PlanBasedChecker - verifies protocol compatibility
        checker = PlanBasedChecker(plan_type=Plan)
        hook = create_task_completion_stop_hook(context, checker=checker)
        input_data = {"hook_event_name": "Stop", "stopReason": "end_turn"}

        # Should work with checker protocol
        result = asyncio.run(hook(input_data, None, {"signal": None}))

        # No plan initialized, so stop is allowed
        assert result == {}
        assert context.stop_reason == "end_turn"

    def test_skips_check_when_deadline_exceeded(self, session: Session) -> None:
        """Stop hook skips task completion check when deadline expired."""
        from weakincentives.clock import FakeClock

        _initialize_plan_session(session)
        # Create incomplete plan
        session.dispatch(
            Plan(
                objective="Test",
                status="active",
                steps=(PlanStep(step_id=1, title="Incomplete", status="pending"),),
            )
        )

        # Create expired deadline using fake clock
        clock = FakeClock()
        anchor = datetime.now(UTC)
        clock.set_wall(anchor)
        expired_deadline = Deadline(anchor + timedelta(seconds=5), clock=clock)
        clock.advance(10)  # Now expired

        constraints = HookConstraints(deadline=expired_deadline)
        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
            constraints=constraints,
        )

        hook = create_task_completion_stop_hook(
            context, checker=PlanBasedChecker(plan_type=Plan)
        )
        input_data = {"hook_event_name": "Stop", "stopReason": "end_turn"}

        # Even with incomplete tasks, stop is allowed due to expired deadline
        result = asyncio.run(hook(input_data, None, {"signal": None}))

        assert result == {}

    def test_skips_check_when_budget_exhausted(self, session: Session) -> None:
        """Stop hook skips task completion check when budget exhausted."""
        _initialize_plan_session(session)
        # Create incomplete plan
        session.dispatch(
            Plan(
                objective="Test",
                status="active",
                steps=(PlanStep(step_id=1, title="Incomplete", status="pending"),),
            )
        )

        # Create exhausted budget tracker
        budget = Budget(max_total_tokens=1000)
        tracker = BudgetTracker(budget=budget)
        tracker.record_cumulative(
            "eval-1", TokenUsage(input_tokens=500, output_tokens=600)
        )  # Over budget

        constraints = HookConstraints(budget_tracker=tracker)
        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
            constraints=constraints,
        )

        hook = create_task_completion_stop_hook(
            context, checker=PlanBasedChecker(plan_type=Plan)
        )
        input_data = {"hook_event_name": "Stop", "stopReason": "end_turn"}

        # Even with incomplete tasks, stop is allowed due to exhausted budget
        result = asyncio.run(hook(input_data, None, {"signal": None}))

        assert result == {}


class TestHookEventNameValidation:
    """Tests for early return when hooks receive wrong event names."""

    def _make_context(self) -> HookContext:
        """Create a basic hook context for testing."""
        prompt = _make_prompt()
        session = Session(dispatcher=InProcessDispatcher())
        return HookContext(
            prompt=cast("PromptProtocol[object]", prompt),
            prompt_name="test-prompt",
            session=session,
            adapter_name="claude_agent_sdk",
        )

    def test_pre_tool_use_hook_wrong_event_name(self) -> None:
        """PreToolUse hook returns empty dict for wrong event name."""
        context = self._make_context()
        hook = create_pre_tool_use_hook(context)

        # Call with wrong event name
        input_data = {"hook_event_name": "PostToolUse", "tool_name": "Read"}
        result = asyncio.run(hook(input_data, None, {"signal": None}))

        assert result == {}

    def test_post_tool_use_hook_wrong_event_name(self) -> None:
        """PostToolUse hook returns empty dict for wrong event name."""
        context = self._make_context()
        hook = create_post_tool_use_hook(context)

        # Call with wrong event name
        input_data = {"hook_event_name": "PreToolUse", "tool_name": "Read"}
        result = asyncio.run(hook(input_data, None, {"signal": None}))

        assert result == {}

    def test_user_prompt_submit_hook_wrong_event_name(self) -> None:
        """UserPromptSubmit hook returns empty dict for wrong event name."""
        context = self._make_context()
        hook = create_user_prompt_submit_hook(context)

        # Call with wrong event name
        input_data = {"hook_event_name": "Stop", "prompt": "test"}
        result = asyncio.run(hook(input_data, None, {"signal": None}))

        assert result == {}

    def test_stop_hook_wrong_event_name(self) -> None:
        """Stop hook returns empty dict for wrong event name."""
        context = self._make_context()
        hook = create_stop_hook(context)

        # Call with wrong event name
        input_data = {"hook_event_name": "PreToolUse", "tool_name": "test"}
        result = asyncio.run(hook(input_data, None, {"signal": None}))

        assert result == {}

    def test_task_completion_stop_hook_wrong_event_name(self) -> None:
        """TaskCompletion stop hook returns empty dict for wrong event name."""
        context = self._make_context()
        hook = create_task_completion_stop_hook(
            context, checker=PlanBasedChecker(plan_type=Plan)
        )

        # Call with wrong event name
        input_data = {"hook_event_name": "PreCompact", "session_id": "test"}
        result = asyncio.run(hook(input_data, None, {"signal": None}))

        assert result == {}

    def test_subagent_stop_hook_wrong_event_name(self) -> None:
        """SubagentStop hook returns empty dict for wrong event name."""
        context = self._make_context()
        hook = create_subagent_stop_hook(context)

        # Call with wrong event name
        input_data = {"hook_event_name": "Stop", "session_id": "test"}
        result = asyncio.run(hook(input_data, None, {"signal": None}))

        assert result == {}

    def test_pre_compact_hook_wrong_event_name(self) -> None:
        """PreCompact hook returns empty dict for wrong event name."""
        context = self._make_context()
        hook = create_pre_compact_hook(context)

        # Call with wrong event name
        input_data = {"hook_event_name": "Stop", "session_id": "test"}
        result = asyncio.run(hook(input_data, None, {"signal": None}))

        assert result == {}

    def test_pre_tool_use_sets_in_subagent_for_task_tool(self) -> None:
        """PreToolUse hook sets in_subagent flag when Task tool is called."""
        context = self._make_context()
        hook = create_pre_tool_use_hook(context)

        # Verify flag is initially False
        assert context.stats.in_subagent is False

        # Call with Task tool
        input_data = {
            "hook_event_name": "PreToolUse",
            "tool_name": "Task",
            "tool_input": {"prompt": "test subagent"},
        }
        asyncio.run(hook(input_data, "tool-123", {"signal": None}))

        # Flag should be set
        assert context.stats.in_subagent is True

    def test_pre_tool_use_does_not_set_in_subagent_for_other_tools(self) -> None:
        """PreToolUse hook does not set in_subagent for non-Task tools."""
        context = self._make_context()
        hook = create_pre_tool_use_hook(context)

        # Verify flag is initially False
        assert context.stats.in_subagent is False

        # Call with regular tool
        input_data = {
            "hook_event_name": "PreToolUse",
            "tool_name": "Read",
            "tool_input": {"file_path": "/test"},
        }
        asyncio.run(hook(input_data, "tool-123", {"signal": None}))

        # Flag should remain False
        assert context.stats.in_subagent is False

    def test_subagent_stop_increments_count_and_clears_flag(self) -> None:
        """SubagentStop hook increments subagent_count and clears in_subagent."""
        context = self._make_context()
        hook = create_subagent_stop_hook(context)

        # Set up initial state as if subagent was running
        context.stats.in_subagent = True
        context.stats.subagent_count = 0

        # Call SubagentStop
        input_data = {
            "hook_event_name": "SubagentStop",
            "session_id": "subagent-123",
            "stop_hook_active": False,
        }
        asyncio.run(hook(input_data, None, {"signal": None}))

        # Flag should be cleared and count incremented
        assert context.stats.in_subagent is False
        assert context.stats.subagent_count == 1

    def test_subagent_lifecycle_flag_management(self) -> None:
        """Test full subagent lifecycle: PreToolUse(Task) -> SubagentStop."""
        context = self._make_context()
        pre_hook = create_pre_tool_use_hook(context)
        subagent_stop_hook = create_subagent_stop_hook(context)

        # Initially not in subagent
        assert context.stats.in_subagent is False
        assert context.stats.subagent_count == 0

        # PreToolUse for Task sets flag
        task_input = {
            "hook_event_name": "PreToolUse",
            "tool_name": "Task",
            "tool_input": {"prompt": "run subagent"},
        }
        asyncio.run(pre_hook(task_input, "task-tool-1", {"signal": None}))
        assert context.stats.in_subagent is True

        # SubagentStop clears flag and increments count
        stop_input = {
            "hook_event_name": "SubagentStop",
            "session_id": "subagent-1",
            "stop_hook_active": False,
        }
        asyncio.run(subagent_stop_hook(stop_input, None, {"signal": None}))
        assert context.stats.in_subagent is False
        assert context.stats.subagent_count == 1


class TestPlanBasedCheckerNoPlanType:
    """Tests for PlanBasedChecker with plan_type=None."""

    def test_no_plan_type_returns_ok(self, session: Session) -> None:
        """Checker returns ok when plan_type is None."""
        from weakincentives.adapters.claude_agent_sdk._task_completion import (
            PlanBasedChecker,
            TaskCompletionContext,
        )

        checker = PlanBasedChecker(plan_type=None)
        context = TaskCompletionContext(
            session=session,
            tentative_output=None,
            stop_reason="end_turn",
        )

        result = checker.check(context)

        assert result.complete is True
        assert "No planning tools available" in result.feedback


class TestCompositeChecker:
    """Tests for CompositeChecker."""

    def test_empty_checkers_returns_ok(self, session: Session) -> None:
        """Empty composite checker returns ok."""
        from weakincentives.adapters.claude_agent_sdk._task_completion import (
            CompositeChecker,
            TaskCompletionContext,
        )

        checker = CompositeChecker(checkers=())
        context = TaskCompletionContext(
            session=session,
            tentative_output=None,
            stop_reason="end_turn",
        )

        result = checker.check(context)

        assert result.complete is True
        assert "No checkers configured" in result.feedback

    def test_all_must_pass_short_circuits_on_failure(self, session: Session) -> None:
        """Composite checker short-circuits on first failure when all_must_pass=True."""
        from weakincentives.adapters.claude_agent_sdk._task_completion import (
            CompositeChecker,
            PlanBasedChecker,
            TaskCompletionContext,
        )

        _initialize_plan_session(session)
        session.dispatch(
            Plan(
                objective="Test",
                status="active",
                steps=(PlanStep(step_id=1, title="Incomplete", status="pending"),),
            )
        )

        checker = CompositeChecker(
            checkers=(
                PlanBasedChecker(plan_type=Plan),
                PlanBasedChecker(plan_type=None),  # Would pass but won't be reached
            ),
            all_must_pass=True,
        )
        context = TaskCompletionContext(
            session=session,
            tentative_output=None,
            stop_reason="end_turn",
        )

        result = checker.check(context)

        assert result.complete is False
        assert "incomplete" in result.feedback.lower()

    def test_all_must_pass_all_succeed(self, session: Session) -> None:
        """Composite checker returns ok when all checkers pass."""
        from weakincentives.adapters.claude_agent_sdk._task_completion import (
            CompositeChecker,
            PlanBasedChecker,
            TaskCompletionContext,
        )

        checker = CompositeChecker(
            checkers=(
                PlanBasedChecker(plan_type=None),
                PlanBasedChecker(plan_type=None),
            ),
            all_must_pass=True,
        )
        context = TaskCompletionContext(
            session=session,
            tentative_output=None,
            stop_reason="end_turn",
        )

        result = checker.check(context)

        assert result.complete is True

    def test_any_pass_short_circuits_on_success(self, session: Session) -> None:
        """Composite checker short-circuits on first success when all_must_pass=False."""
        from weakincentives.adapters.claude_agent_sdk._task_completion import (
            CompositeChecker,
            PlanBasedChecker,
            TaskCompletionContext,
        )

        _initialize_plan_session(session)
        session.dispatch(
            Plan(
                objective="Test",
                status="active",
                steps=(PlanStep(step_id=1, title="Incomplete", status="pending"),),
            )
        )

        checker = CompositeChecker(
            checkers=(
                PlanBasedChecker(plan_type=None),  # Passes
                PlanBasedChecker(plan_type=Plan),  # Would fail but won't be reached
            ),
            all_must_pass=False,
        )
        context = TaskCompletionContext(
            session=session,
            tentative_output=None,
            stop_reason="end_turn",
        )

        result = checker.check(context)

        assert result.complete is True

    def test_any_pass_all_fail(self, session: Session) -> None:
        """Composite checker returns incomplete when all checkers fail in any mode."""
        from weakincentives.adapters.claude_agent_sdk._task_completion import (
            CompositeChecker,
            PlanBasedChecker,
            TaskCompletionContext,
        )

        _initialize_plan_session(session)
        session.dispatch(
            Plan(
                objective="Test",
                status="active",
                steps=(PlanStep(step_id=1, title="Incomplete", status="pending"),),
            )
        )

        checker = CompositeChecker(
            checkers=(
                PlanBasedChecker(plan_type=Plan),
                PlanBasedChecker(plan_type=Plan),
            ),
            all_must_pass=False,
        )
        context = TaskCompletionContext(
            session=session,
            tentative_output=None,
            stop_reason="end_turn",
        )

        result = checker.check(context)

        assert result.complete is False
