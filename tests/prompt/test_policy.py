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

"""Tests for tool policy enforcement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from weakincentives.contrib.tools import InMemoryFilesystem
from weakincentives.filesystem import Filesystem
from weakincentives.prompt import (
    PolicyDecision,
    PolicyState,
    Prompt,
    PromptTemplate,
    ReadBeforeWritePolicy,
    SequentialDependencyPolicy,
    Tool,
    ToolContext,
    ToolResult,
)
from weakincentives.prompt.policy import _extract_path
from weakincentives.resources import Binding, ResourceRegistry
from weakincentives.runtime import InProcessDispatcher, Session

if TYPE_CHECKING:
    from weakincentives.runtime.logging import StructuredLogger


# --- Test Parameters ---


@dataclass(frozen=True)
class FileParams:
    path: str


@dataclass(frozen=True)
class FilePathParams:
    file_path: str


@dataclass(frozen=True)
class NoPathParams:
    value: str


# --- PolicyDecision Tests ---


class TestPolicyDecision:
    def test_allow_creates_allowed_decision(self) -> None:
        decision = PolicyDecision.allow()
        assert decision.allowed is True
        assert decision.reason is None

    def test_deny_creates_denied_decision_with_reason(self) -> None:
        decision = PolicyDecision.deny("file not read")
        assert decision.allowed is False
        assert decision.reason == "file not read"

    def test_decision_is_frozen(self) -> None:
        decision = PolicyDecision.allow()
        with pytest.raises(AttributeError):
            decision.allowed = False  # type: ignore[misc]


# --- PolicyState Tests ---


class TestPolicyState:
    def test_default_state_has_empty_sets(self) -> None:
        state = PolicyState(policy_name="test")
        assert state.invoked_tools == frozenset()
        assert state.invoked_keys == frozenset()

    def test_state_with_invoked_tools(self) -> None:
        state = PolicyState(
            policy_name="test", invoked_tools=frozenset({"read_file", "list_files"})
        )
        assert "read_file" in state.invoked_tools
        assert "list_files" in state.invoked_tools

    def test_state_with_invoked_keys(self) -> None:
        state = PolicyState(
            policy_name="test",
            invoked_keys=frozenset(
                {("read_file", "/path/a"), ("read_file", "/path/b")}
            ),
        )
        assert ("read_file", "/path/a") in state.invoked_keys
        assert ("read_file", "/path/b") in state.invoked_keys


# --- _extract_path Tests ---


class TestExtractPath:
    def test_extracts_path_field(self) -> None:
        params = FileParams(path="/test/file.txt")
        assert _extract_path(params) == "/test/file.txt"

    def test_extracts_file_path_field(self) -> None:
        params = FilePathParams(file_path="/test/other.txt")
        assert _extract_path(params) == "/test/other.txt"

    def test_returns_none_for_no_path_field(self) -> None:
        params = NoPathParams(value="test")
        assert _extract_path(params) is None

    def test_returns_none_for_none_params(self) -> None:
        assert _extract_path(None) is None

    def test_returns_none_for_non_string_path(self) -> None:
        @dataclass(frozen=True)
        class IntPath:
            path: int

        params = IntPath(path=123)
        assert _extract_path(params) is None


# --- SequentialDependencyPolicy Tests ---


def _noop_handler(params: None, *, context: ToolContext) -> ToolResult[None]:
    """No-op handler for testing."""
    return ToolResult.ok(None)


class TestSequentialDependencyPolicy:
    def _make_context(self, session: Session) -> ToolContext:
        """Create a minimal ToolContext for testing."""
        template: PromptTemplate[None] = PromptTemplate(
            ns="test", key="test-prompt", name="test"
        )
        prompt: Prompt[None] = Prompt(template)
        return ToolContext(
            prompt=prompt,
            rendered_prompt=None,  # type: ignore[arg-type]
            adapter=None,  # type: ignore[arg-type]
            session=session,
            deadline=None,
        )

    def _make_tool(self, name: str) -> Tool[None, None]:
        """Create a minimal tool for testing."""
        return Tool[None, None](
            name=name, description=f"Tool {name}", handler=_noop_handler
        )

    def test_name_property(self) -> None:
        policy = SequentialDependencyPolicy(dependencies={})
        assert policy.name == "sequential_dependency"

    def test_allows_tool_without_dependencies(self) -> None:
        policy = SequentialDependencyPolicy(
            dependencies={"deploy": frozenset({"build"})}
        )
        bus = InProcessDispatcher()
        session = Session(bus=bus)
        context = self._make_context(session)
        tool = self._make_tool("unrelated_tool")

        decision = policy.check(tool, None, context=context)
        assert decision.allowed is True

    def test_denies_tool_when_dependency_not_invoked(self) -> None:
        policy = SequentialDependencyPolicy(
            dependencies={"deploy": frozenset({"build", "test"})}
        )
        bus = InProcessDispatcher()
        session = Session(bus=bus)
        context = self._make_context(session)
        tool = self._make_tool("deploy")

        decision = policy.check(tool, None, context=context)
        assert decision.allowed is False
        assert decision.reason is not None
        assert "deploy" in decision.reason
        assert "build" in decision.reason or "test" in decision.reason

    def test_allows_tool_when_dependencies_satisfied(self) -> None:
        policy = SequentialDependencyPolicy(
            dependencies={"deploy": frozenset({"build"})}
        )
        bus = InProcessDispatcher()
        session = Session(bus=bus)
        # Seed state with build invoked
        session[PolicyState].seed(
            PolicyState(policy_name="test", invoked_tools=frozenset({"build"}))
        )
        context = self._make_context(session)
        tool = self._make_tool("deploy")

        decision = policy.check(tool, None, context=context)
        assert decision.allowed is True

    def test_on_result_records_successful_invocation(self) -> None:
        policy = SequentialDependencyPolicy(dependencies={})
        bus = InProcessDispatcher()
        session = Session(bus=bus)
        context = self._make_context(session)
        tool = self._make_tool("build")
        result: ToolResult[None] = ToolResult.ok(None, message="success")

        policy.on_result(tool, None, result, context=context)

        state = session[PolicyState].latest()
        assert state is not None
        assert "build" in state.invoked_tools

    def test_on_result_does_not_record_failed_invocation(self) -> None:
        policy = SequentialDependencyPolicy(dependencies={})
        bus = InProcessDispatcher()
        session = Session(bus=bus)
        context = self._make_context(session)
        tool = self._make_tool("build")
        result: ToolResult[None] = ToolResult.error("failed")

        policy.on_result(tool, None, result, context=context)

        state = session[PolicyState].latest()
        assert state is None

    def test_on_result_preserves_existing_state(self) -> None:
        policy = SequentialDependencyPolicy(dependencies={})
        bus = InProcessDispatcher()
        session = Session(bus=bus)
        session[PolicyState].seed(
            PolicyState(
                policy_name="test",
                invoked_tools=frozenset({"lint"}),
                invoked_keys=frozenset({("read_file", "/x")}),
            )
        )
        context = self._make_context(session)
        tool = self._make_tool("build")
        result: ToolResult[None] = ToolResult.ok(None)

        policy.on_result(tool, None, result, context=context)

        state = session[PolicyState].latest()
        assert state is not None
        assert "lint" in state.invoked_tools
        assert "build" in state.invoked_tools
        assert ("read_file", "/x") in state.invoked_keys


# --- ReadBeforeWritePolicy Tests ---


def _file_handler(params: FileParams, *, context: ToolContext) -> ToolResult[None]:
    """File handler for testing."""
    return ToolResult.ok(None)


class TestReadBeforeWritePolicy:
    def _make_context(
        self, session: Session, *, filesystem: InMemoryFilesystem | None = None
    ) -> ToolContext:
        """Create a ToolContext with optional filesystem."""
        if filesystem is not None:
            # Register with Filesystem protocol as key
            registry = ResourceRegistry.of(
                Binding(Filesystem, lambda _: filesystem)  # type: ignore[type-abstract]
            )
            template: PromptTemplate[None] = PromptTemplate(
                ns="test", key="test-prompt", name="test", resources=registry
            )
            prompt: Prompt[None] = Prompt(template)
            prompt = prompt.bind(resources={Filesystem: filesystem})  # type: ignore[type-abstract]
        else:
            template = PromptTemplate(ns="test", key="test-prompt", name="test")
            prompt = Prompt(template)

        # Always enter resource context
        prompt.resources.__enter__()

        return ToolContext(
            prompt=prompt,
            rendered_prompt=None,  # type: ignore[arg-type]
            adapter=None,  # type: ignore[arg-type]
            session=session,
            deadline=None,
        )

    def _make_tool(self, name: str) -> Tool[FileParams, None]:
        """Create a tool with path parameter."""
        return Tool[FileParams, None](
            name=name, description=f"Tool {name}", handler=_file_handler
        )

    def test_name_property(self) -> None:
        policy = ReadBeforeWritePolicy()
        assert policy.name == "read_before_write"

    def test_allows_non_write_tool(self) -> None:
        policy = ReadBeforeWritePolicy()
        bus = InProcessDispatcher()
        session = Session(bus=bus)
        context = self._make_context(session)
        tool = self._make_tool("list_files")

        decision = policy.check(tool, FileParams(path="/x"), context=context)
        assert decision.allowed is True

    def test_allows_write_when_no_path(self) -> None:
        policy = ReadBeforeWritePolicy()
        bus = InProcessDispatcher()
        session = Session(bus=bus)
        context = self._make_context(session)
        tool = Tool[NoPathParams, None](
            name="write_file",
            description="Write file",
            handler=None,
        )

        decision = policy.check(tool, NoPathParams(value="x"), context=context)
        assert decision.allowed is True

    def test_allows_write_when_no_filesystem(self) -> None:
        policy = ReadBeforeWritePolicy()
        bus = InProcessDispatcher()
        session = Session(bus=bus)
        context = self._make_context(session, filesystem=None)
        tool = self._make_tool("write_file")

        decision = policy.check(tool, FileParams(path="/x"), context=context)
        assert decision.allowed is True

    def test_allows_new_file_creation(self) -> None:
        fs = InMemoryFilesystem()
        # File does not exist
        policy = ReadBeforeWritePolicy()
        bus = InProcessDispatcher()
        session = Session(bus=bus)
        context = self._make_context(session, filesystem=fs)
        tool = self._make_tool("write_file")

        decision = policy.check(tool, FileParams(path="/new.txt"), context=context)
        assert decision.allowed is True

    def test_denies_overwrite_without_read(self) -> None:
        fs = InMemoryFilesystem()
        fs.write("/existing.txt", "content")

        policy = ReadBeforeWritePolicy()
        bus = InProcessDispatcher()
        session = Session(bus=bus)
        context = self._make_context(session, filesystem=fs)
        tool = self._make_tool("write_file")

        decision = policy.check(tool, FileParams(path="/existing.txt"), context=context)
        assert decision.allowed is False
        assert decision.reason is not None
        assert "/existing.txt" in decision.reason
        assert "read" in decision.reason.lower()

    def test_allows_overwrite_after_read(self) -> None:
        fs = InMemoryFilesystem()
        fs.write("/existing.txt", "content")

        policy = ReadBeforeWritePolicy()
        bus = InProcessDispatcher()
        session = Session(bus=bus)
        # Record that file was read
        session[PolicyState].seed(
            PolicyState(
                policy_name="test",
                invoked_keys=frozenset({("read_file", "/existing.txt")}),
            )
        )
        context = self._make_context(session, filesystem=fs)
        tool = self._make_tool("write_file")

        decision = policy.check(tool, FileParams(path="/existing.txt"), context=context)
        assert decision.allowed is True

    def test_on_result_records_read_operation(self) -> None:
        policy = ReadBeforeWritePolicy()
        bus = InProcessDispatcher()
        session = Session(bus=bus)
        context = self._make_context(session)
        tool = self._make_tool("read_file")
        result: ToolResult[None] = ToolResult.ok(None)

        policy.on_result(tool, FileParams(path="/test.txt"), result, context=context)

        state = session[PolicyState].latest()
        assert state is not None
        assert ("read_file", "/test.txt") in state.invoked_keys

    def test_on_result_does_not_record_failed_read(self) -> None:
        policy = ReadBeforeWritePolicy()
        bus = InProcessDispatcher()
        session = Session(bus=bus)
        context = self._make_context(session)
        tool = self._make_tool("read_file")
        result: ToolResult[None] = ToolResult.error("failed")

        policy.on_result(tool, FileParams(path="/test.txt"), result, context=context)

        state = session[PolicyState].latest()
        assert state is None

    def test_on_result_does_not_record_non_read_tool(self) -> None:
        policy = ReadBeforeWritePolicy()
        bus = InProcessDispatcher()
        session = Session(bus=bus)
        context = self._make_context(session)
        tool = self._make_tool("write_file")
        result: ToolResult[None] = ToolResult.ok(None)

        policy.on_result(tool, FileParams(path="/test.txt"), result, context=context)

        state = session[PolicyState].latest()
        assert state is None

    def test_on_result_does_not_record_when_no_path(self) -> None:
        policy = ReadBeforeWritePolicy()
        bus = InProcessDispatcher()
        session = Session(bus=bus)
        context = self._make_context(session)
        tool = Tool[NoPathParams, None](
            name="read_file",
            description="Read",
            handler=None,
        )
        result: ToolResult[None] = ToolResult.ok(None)

        policy.on_result(tool, NoPathParams(value="x"), result, context=context)

        state = session[PolicyState].latest()
        assert state is None

    def test_on_result_preserves_existing_state(self) -> None:
        policy = ReadBeforeWritePolicy()
        bus = InProcessDispatcher()
        session = Session(bus=bus)
        session[PolicyState].seed(
            PolicyState(
                policy_name="test",
                invoked_tools=frozenset({"lint"}),
                invoked_keys=frozenset({("read_file", "/x")}),
            )
        )
        context = self._make_context(session)
        tool = self._make_tool("read_file")
        result: ToolResult[None] = ToolResult.ok(None)

        policy.on_result(tool, FileParams(path="/y"), result, context=context)

        state = session[PolicyState].latest()
        assert state is not None
        assert "lint" in state.invoked_tools
        assert ("read_file", "/x") in state.invoked_keys
        assert ("read_file", "/y") in state.invoked_keys

    def test_custom_read_write_tools(self) -> None:
        policy = ReadBeforeWritePolicy(
            read_tools=frozenset({"fetch_file"}),
            write_tools=frozenset({"save_file"}),
        )
        fs = InMemoryFilesystem()
        fs.write("/data.json", "{}")

        bus = InProcessDispatcher()
        session = Session(bus=bus)
        context = self._make_context(session, filesystem=fs)

        # Standard write_file should be allowed (not in our write_tools)
        tool1 = self._make_tool("write_file")
        decision1 = policy.check(tool1, FileParams(path="/data.json"), context=context)
        assert decision1.allowed is True

        # Custom save_file should be denied (file not fetched)
        tool2 = self._make_tool("save_file")
        decision2 = policy.check(tool2, FileParams(path="/data.json"), context=context)
        assert decision2.allowed is False

        # After fetch, save should be allowed
        session[PolicyState].seed(
            PolicyState(
                policy_name="test",
                invoked_keys=frozenset({("fetch_file", "/data.json")}),
            )
        )
        decision3 = policy.check(tool2, FileParams(path="/data.json"), context=context)
        assert decision3.allowed is True


# --- Tool Executor Policy Integration Tests ---


class TestToolExecutorPolicyHelpers:
    """Test policy helper functions used by ToolExecutor."""

    def _make_context(self, session: Session) -> ToolContext:
        """Create a minimal ToolContext for testing."""
        template: PromptTemplate[None] = PromptTemplate(
            ns="test", key="test-prompt", name="test"
        )
        prompt: Prompt[None] = Prompt(template)
        prompt.resources.__enter__()
        return ToolContext(
            prompt=prompt,
            rendered_prompt=None,  # type: ignore[arg-type]
            adapter=None,  # type: ignore[arg-type]
            session=session,
            deadline=None,
        )

    def _make_logger(self) -> StructuredLogger:
        """Create a mock logger for testing."""
        import logging

        from weakincentives.runtime.logging import StructuredLogger

        logger = logging.getLogger("test_policy")
        logger.addHandler(logging.NullHandler())
        return StructuredLogger(logger)

    def test_check_policies_returns_allow_for_empty_policies(self) -> None:
        from weakincentives.adapters.tool_executor import _check_policies

        bus = InProcessDispatcher()
        session = Session(bus=bus)
        context = self._make_context(session)
        tool = Tool[FileParams, None](
            name="test_tool", description="Test", handler=None
        )
        log = self._make_logger()

        decision = _check_policies(
            policies=(),
            tool=tool,
            tool_params=FileParams(path="/x"),
            context=context,
            log=log,
        )
        assert decision.allowed is True

    def test_check_policies_returns_deny_from_first_denying_policy(self) -> None:
        from weakincentives.adapters.tool_executor import _check_policies

        bus = InProcessDispatcher()
        session = Session(bus=bus)
        context = self._make_context(session)
        tool = Tool[FileParams, None](name="deploy", description="Deploy", handler=None)
        log = self._make_logger()

        # Policy that requires "build" to be invoked first
        policy = SequentialDependencyPolicy(
            dependencies={"deploy": frozenset({"build"})}
        )

        decision = _check_policies(
            policies=(policy,),
            tool=tool,
            tool_params=FileParams(path="/x"),
            context=context,
            log=log,
        )
        assert decision.allowed is False
        assert "build" in (decision.reason or "")

    def test_check_policies_iterates_through_all_policies(self) -> None:
        from weakincentives.adapters.tool_executor import _check_policies

        bus = InProcessDispatcher()
        session = Session(bus=bus)
        context = self._make_context(session)
        tool = Tool[FileParams, None](
            name="unrelated", description="Unrelated tool", handler=None
        )
        log = self._make_logger()

        # Multiple policies where both allow (no dependencies for this tool)
        policy1 = SequentialDependencyPolicy(
            dependencies={"deploy": frozenset({"build"})}
        )
        policy2 = SequentialDependencyPolicy(
            dependencies={"release": frozenset({"deploy"})}
        )

        decision = _check_policies(
            policies=(policy1, policy2),
            tool=tool,
            tool_params=FileParams(path="/x"),
            context=context,
            log=log,
        )
        # Both policies should allow since "unrelated" has no dependencies
        assert decision.allowed is True

    def test_notify_policies_of_result_skips_failed_results(self) -> None:
        from weakincentives.adapters.tool_executor import _notify_policies_of_result

        bus = InProcessDispatcher()
        session = Session(bus=bus)
        context = self._make_context(session)
        tool = Tool[FileParams, None](
            name="read_file", description="Read", handler=None
        )
        policy = ReadBeforeWritePolicy()
        result: ToolResult[None] = ToolResult.error("failed")

        _notify_policies_of_result(
            policies=(policy,),
            tool=tool,
            tool_params=FileParams(path="/test.txt"),
            result=result,
            context=context,
        )

        # State should not be updated for failed results
        state = session[PolicyState].latest()
        assert state is None

    def test_notify_policies_of_result_calls_on_result_for_success(self) -> None:
        from weakincentives.adapters.tool_executor import _notify_policies_of_result

        bus = InProcessDispatcher()
        session = Session(bus=bus)
        context = self._make_context(session)
        tool = Tool[FileParams, None](
            name="read_file", description="Read", handler=None
        )
        policy = ReadBeforeWritePolicy()
        result: ToolResult[None] = ToolResult.ok(None)

        _notify_policies_of_result(
            policies=(policy,),
            tool=tool,
            tool_params=FileParams(path="/test.txt"),
            result=result,
            context=context,
        )

        # State should be updated for successful read
        state = session[PolicyState].latest()
        assert state is not None
        assert ("read_file", "/test.txt") in state.invoked_keys

    def test_build_policy_denied_result_creates_error_result(self) -> None:
        from weakincentives.adapters.tool_executor import _build_policy_denied_result

        decision = PolicyDecision.deny("file not read")
        result = _build_policy_denied_result(decision)

        assert result.success is False
        assert result.message == "file not read"
        assert result.value is None

    def test_build_policy_denied_result_uses_default_message(self) -> None:
        from weakincentives.adapters.tool_executor import _build_policy_denied_result

        decision = PolicyDecision(allowed=False, reason=None)
        result = _build_policy_denied_result(decision)

        assert result.success is False
        assert result.message == "Policy denied"
