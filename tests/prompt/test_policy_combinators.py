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

"""Tests for policy combinators and state isolation."""

from __future__ import annotations

from dataclasses import dataclass

from weakincentives.contrib.tools import InMemoryFilesystem
from weakincentives.filesystem import Filesystem
from weakincentives.prompt import (
    AllOfPolicy,
    AnyOfPolicy,
    Prompt,
    PromptTemplate,
    ReadBeforeWritePolicy,
    ReadBeforeWriteState,
    SequentialDependencyPolicy,
    SequentialDependencyState,
    Tool,
    ToolContext,
    ToolResult,
)
from weakincentives.resources import Binding, ResourceRegistry
from weakincentives.runtime import InProcessDispatcher, Session

# --- Test Parameters ---


@dataclass(frozen=True)
class FileParams:
    path: str


# --- Shared Handlers ---


def _noop_handler(params: None, *, context: ToolContext) -> ToolResult[None]:
    """No-op handler for testing."""
    return ToolResult.ok(None)


def _file_handler(params: FileParams, *, context: ToolContext) -> ToolResult[None]:
    """File handler for testing."""
    return ToolResult.ok(None)


# --- State Isolation Tests ---


class TestPolicyStateIsolation:
    """Verify that policies use independent session slices."""

    def _make_context(self, session: Session) -> ToolContext:
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

    def _make_context_with_fs(
        self, session: Session, filesystem: InMemoryFilesystem
    ) -> ToolContext:
        registry = ResourceRegistry.of(
            Binding(Filesystem, lambda _: filesystem)  # type: ignore[type-abstract]
        )
        template: PromptTemplate[None] = PromptTemplate(
            ns="test", key="test-prompt", name="test", resources=registry
        )
        prompt: Prompt[None] = Prompt(template)
        prompt = prompt.bind(resources={Filesystem: filesystem})  # type: ignore[type-abstract]
        prompt.resources.__enter__()
        return ToolContext(
            prompt=prompt,
            rendered_prompt=None,  # type: ignore[arg-type]
            adapter=None,  # type: ignore[arg-type]
            session=session,
            deadline=None,
        )

    def test_sequential_and_rbw_do_not_interfere(self) -> None:
        """Two policies writing to the same session don't clobber state."""
        seq_policy = SequentialDependencyPolicy(dependencies={})
        rbw_policy = ReadBeforeWritePolicy()

        dispatcher = InProcessDispatcher()
        session = Session(dispatcher=dispatcher)
        context = self._make_context(session)

        # SequentialDependency records a tool invocation
        build_tool = Tool[None, None](
            name="build", description="Build", handler=_noop_handler
        )
        seq_policy.on_result(build_tool, None, ToolResult.ok(None), context=context)

        # ReadBeforeWrite records a read
        read_tool = Tool[FileParams, None](
            name="read_file", description="Read", handler=_file_handler
        )
        rbw_policy.on_result(
            read_tool,
            FileParams(path="/config.yaml"),
            ToolResult.ok(None),
            context=context,
        )

        # Verify each policy's state is independent
        seq_state = session[SequentialDependencyState].latest()
        assert seq_state is not None
        assert "build" in seq_state.invoked_tools

        rbw_state = session[ReadBeforeWriteState].latest()
        assert rbw_state is not None
        assert ("read_file", "config.yaml") in rbw_state.invoked_keys

    def test_sequential_state_unaffected_by_rbw_seed(self) -> None:
        """Seeding ReadBeforeWriteState doesn't clear SequentialDependencyState."""
        dispatcher = InProcessDispatcher()
        session = Session(dispatcher=dispatcher)

        session[SequentialDependencyState].seed(
            SequentialDependencyState(invoked_tools=frozenset({"lint"}))
        )
        session[ReadBeforeWriteState].seed(
            ReadBeforeWriteState(invoked_keys=frozenset({("read_file", "x")}))
        )

        # Both states should coexist
        seq_state = session[SequentialDependencyState].latest()
        assert seq_state is not None
        assert "lint" in seq_state.invoked_tools

        rbw_state = session[ReadBeforeWriteState].latest()
        assert rbw_state is not None
        assert ("read_file", "x") in rbw_state.invoked_keys

    def test_policies_work_together_full_flow(self) -> None:
        """Both policies enforce correctly when used on the same session."""
        seq_policy = SequentialDependencyPolicy(
            dependencies={"deploy": frozenset({"build"})}
        )
        rbw_policy = ReadBeforeWritePolicy()

        fs = InMemoryFilesystem()
        fs.write("config.yaml", "content")

        dispatcher = InProcessDispatcher()
        session = Session(dispatcher=dispatcher)
        context = self._make_context_with_fs(session, fs)

        # Sequential: deploy should be denied (build not invoked)
        deploy_tool = Tool[None, None](
            name="deploy", description="Deploy", handler=_noop_handler
        )
        decision1 = seq_policy.check(deploy_tool, None, context=context)
        assert decision1.allowed is False

        # RBW: write should be denied (config.yaml not read)
        write_tool = Tool[FileParams, None](
            name="write_file", description="Write", handler=_file_handler
        )
        decision2 = rbw_policy.check(
            write_tool, FileParams(path="/config.yaml"), context=context
        )
        assert decision2.allowed is False

        # Record build and read
        build_tool = Tool[None, None](
            name="build", description="Build", handler=_noop_handler
        )
        seq_policy.on_result(build_tool, None, ToolResult.ok(None), context=context)
        read_tool = Tool[FileParams, None](
            name="read_file", description="Read", handler=_file_handler
        )
        rbw_policy.on_result(
            read_tool,
            FileParams(path="/config.yaml"),
            ToolResult.ok(None),
            context=context,
        )

        # Now both should allow
        decision3 = seq_policy.check(deploy_tool, None, context=context)
        assert decision3.allowed is True
        decision4 = rbw_policy.check(
            write_tool, FileParams(path="/config.yaml"), context=context
        )
        assert decision4.allowed is True


# --- AllOfPolicy Tests ---


class TestAllOfPolicy:
    def _make_context(self, session: Session) -> ToolContext:
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

    def test_name_property(self) -> None:
        policy = AllOfPolicy(policies=())
        assert policy.name == "all_of"

    def test_allows_when_all_children_allow(self) -> None:
        policy = AllOfPolicy(
            policies=(
                SequentialDependencyPolicy(dependencies={}),
                SequentialDependencyPolicy(dependencies={}),
            )
        )
        dispatcher = InProcessDispatcher()
        session = Session(dispatcher=dispatcher)
        context = self._make_context(session)
        tool = Tool[None, None](
            name="deploy", description="Deploy", handler=_noop_handler
        )

        decision = policy.check(tool, None, context=context)
        assert decision.allowed is True

    def test_denies_when_any_child_denies(self) -> None:
        allowing = SequentialDependencyPolicy(dependencies={})
        denying = SequentialDependencyPolicy(
            dependencies={"deploy": frozenset({"build"})}
        )
        policy = AllOfPolicy(policies=(allowing, denying))

        dispatcher = InProcessDispatcher()
        session = Session(dispatcher=dispatcher)
        context = self._make_context(session)
        tool = Tool[None, None](
            name="deploy", description="Deploy", handler=_noop_handler
        )

        decision = policy.check(tool, None, context=context)
        assert decision.allowed is False
        assert "build" in (decision.reason or "")

    def test_short_circuits_on_first_denial(self) -> None:
        """First denying policy stops evaluation."""
        denying1 = SequentialDependencyPolicy(
            dependencies={"deploy": frozenset({"build"})}
        )
        denying2 = SequentialDependencyPolicy(
            dependencies={"deploy": frozenset({"test"})}
        )
        policy = AllOfPolicy(policies=(denying1, denying2))

        dispatcher = InProcessDispatcher()
        session = Session(dispatcher=dispatcher)
        context = self._make_context(session)
        tool = Tool[None, None](
            name="deploy", description="Deploy", handler=_noop_handler
        )

        decision = policy.check(tool, None, context=context)
        assert decision.allowed is False
        # Should contain build (from first policy) but not test
        assert "build" in (decision.reason or "")

    def test_allows_with_empty_policies(self) -> None:
        policy = AllOfPolicy(policies=())
        dispatcher = InProcessDispatcher()
        session = Session(dispatcher=dispatcher)
        context = self._make_context(session)
        tool = Tool[None, None](
            name="deploy", description="Deploy", handler=_noop_handler
        )

        decision = policy.check(tool, None, context=context)
        assert decision.allowed is True

    def test_on_result_delegates_to_all_children(self) -> None:
        policy1 = SequentialDependencyPolicy(dependencies={})
        policy2 = SequentialDependencyPolicy(dependencies={})
        composite = AllOfPolicy(policies=(policy1, policy2))

        dispatcher = InProcessDispatcher()
        session = Session(dispatcher=dispatcher)
        context = self._make_context(session)
        tool = Tool[None, None](
            name="build", description="Build", handler=_noop_handler
        )
        result: ToolResult[None] = ToolResult.ok(None)

        composite.on_result(tool, None, result, context=context)

        state = session[SequentialDependencyState].latest()
        assert state is not None
        assert "build" in state.invoked_tools


# --- AnyOfPolicy Tests ---


class TestAnyOfPolicy:
    def _make_context(self, session: Session) -> ToolContext:
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

    def test_name_property(self) -> None:
        policy = AnyOfPolicy(policies=())
        assert policy.name == "any_of"

    def test_allows_when_any_child_allows(self) -> None:
        denying = SequentialDependencyPolicy(
            dependencies={"deploy": frozenset({"build"})}
        )
        allowing = SequentialDependencyPolicy(dependencies={})
        policy = AnyOfPolicy(policies=(denying, allowing))

        dispatcher = InProcessDispatcher()
        session = Session(dispatcher=dispatcher)
        context = self._make_context(session)
        tool = Tool[None, None](
            name="deploy", description="Deploy", handler=_noop_handler
        )

        decision = policy.check(tool, None, context=context)
        assert decision.allowed is True

    def test_denies_when_all_children_deny(self) -> None:
        denying1 = SequentialDependencyPolicy(
            dependencies={"deploy": frozenset({"build"})}
        )
        denying2 = SequentialDependencyPolicy(
            dependencies={"deploy": frozenset({"test"})}
        )
        policy = AnyOfPolicy(policies=(denying1, denying2))

        dispatcher = InProcessDispatcher()
        session = Session(dispatcher=dispatcher)
        context = self._make_context(session)
        tool = Tool[None, None](
            name="deploy", description="Deploy", handler=_noop_handler
        )

        decision = policy.check(tool, None, context=context)
        assert decision.allowed is False
        assert decision.reason is not None
        # Combined reasons from both policies
        assert "build" in decision.reason
        assert "test" in decision.reason

    def test_combined_denial_includes_all_suggestions(self) -> None:
        denying1 = SequentialDependencyPolicy(
            dependencies={"deploy": frozenset({"build"})}
        )
        denying2 = SequentialDependencyPolicy(
            dependencies={"deploy": frozenset({"test"})}
        )
        policy = AnyOfPolicy(policies=(denying1, denying2))

        dispatcher = InProcessDispatcher()
        session = Session(dispatcher=dispatcher)
        context = self._make_context(session)
        tool = Tool[None, None](
            name="deploy", description="Deploy", handler=_noop_handler
        )

        decision = policy.check(tool, None, context=context)
        assert decision.allowed is False
        # Should contain suggestions from both child denials
        assert any("build" in s for s in decision.suggestions)
        assert any("test" in s for s in decision.suggestions)

    def test_short_circuits_on_first_allow(self) -> None:
        allowing = SequentialDependencyPolicy(dependencies={})
        denying = SequentialDependencyPolicy(
            dependencies={"deploy": frozenset({"build"})}
        )
        policy = AnyOfPolicy(policies=(allowing, denying))

        dispatcher = InProcessDispatcher()
        session = Session(dispatcher=dispatcher)
        context = self._make_context(session)
        tool = Tool[None, None](
            name="deploy", description="Deploy", handler=_noop_handler
        )

        decision = policy.check(tool, None, context=context)
        assert decision.allowed is True

    def test_denies_with_empty_policies(self) -> None:
        policy = AnyOfPolicy(policies=())
        dispatcher = InProcessDispatcher()
        session = Session(dispatcher=dispatcher)
        context = self._make_context(session)
        tool = Tool[None, None](
            name="deploy", description="Deploy", handler=_noop_handler
        )

        decision = policy.check(tool, None, context=context)
        assert decision.allowed is False
        assert "All policies denied" in (decision.reason or "")

    def test_on_result_delegates_to_all_children(self) -> None:
        policy1 = SequentialDependencyPolicy(dependencies={})
        policy2 = SequentialDependencyPolicy(dependencies={})
        composite = AnyOfPolicy(policies=(policy1, policy2))

        dispatcher = InProcessDispatcher()
        session = Session(dispatcher=dispatcher)
        context = self._make_context(session)
        tool = Tool[None, None](
            name="build", description="Build", handler=_noop_handler
        )
        result: ToolResult[None] = ToolResult.ok(None)

        composite.on_result(tool, None, result, context=context)

        state = session[SequentialDependencyState].latest()
        assert state is not None
        assert "build" in state.invoked_tools
