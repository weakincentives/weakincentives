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

"""Tests for ``weakincentives.policies``."""

from __future__ import annotations

from typing import Any

import pytest

from weakincentives.core import (
    PolicyDeniedError,
    Prompt,
    RenderedPrompt,
    Session,
    Tool,
    ToolContext,
    ToolPolicy,
    ToolResult,
)
from weakincentives.core.prompt import MarkdownSection
from weakincentives.policies import (
    MaxInvocationsPolicy,
    OnceOnlyPolicy,
    ReadBeforeWritePolicy,
    SequentialDependencyPolicy,
)


def _noop(params: object, context: ToolContext) -> ToolResult[None]:
    del params, context
    return ToolResult.ok(None)


def _tool(name: str) -> Tool[Any, Any]:
    return Tool(name=name, description="d", handler=_noop)


def _context() -> ToolContext:
    section = MarkdownSection[None](title="T", key="t")
    prompt = Prompt(ns="ns", key="k", sections=(section,))
    rendered = RenderedPrompt(text="", tools=())
    return ToolContext(session=Session(), prompt=prompt, rendered=rendered)


def test_each_policy_satisfies_protocol() -> None:
    candidates: list[ToolPolicy] = [
        SequentialDependencyPolicy(predecessor="a", successor="b"),
        ReadBeforeWritePolicy(read_tool="r", write_tool="w"),
        MaxInvocationsPolicy.for_tool("a", max_calls=1),
        OnceOnlyPolicy("x"),
    ]
    for policy in candidates:
        assert isinstance(policy, ToolPolicy)


# ---------------------------------------------------------------------------
# SequentialDependencyPolicy
# ---------------------------------------------------------------------------


def test_sequential_dependency_blocks_until_predecessor_succeeds() -> None:
    policy = SequentialDependencyPolicy(predecessor="plan", successor="apply")
    context = _context()

    # Other tools pass through.
    policy.before_invoke(_tool("unrelated"), context)

    # successor blocked first.
    with pytest.raises(PolicyDeniedError, match="apply"):
        policy.before_invoke(_tool("apply"), context)

    # Failed predecessor doesn't unlock.
    policy.on_result(_tool("plan"), ToolResult.error("nope"))
    with pytest.raises(PolicyDeniedError):
        policy.before_invoke(_tool("apply"), context)

    # Successful predecessor unlocks.
    policy.on_result(_tool("plan"), ToolResult.ok(None))
    policy.before_invoke(_tool("apply"), context)


def test_sequential_dependency_ignores_unrelated_results() -> None:
    policy = SequentialDependencyPolicy(predecessor="a", successor="b")
    policy.on_result(_tool("c"), ToolResult.ok(None))
    with pytest.raises(PolicyDeniedError):
        policy.before_invoke(_tool("b"), _context())


# ---------------------------------------------------------------------------
# ReadBeforeWritePolicy
# ---------------------------------------------------------------------------


def test_read_before_write_blocks_writes_until_read_succeeds() -> None:
    policy = ReadBeforeWritePolicy(read_tool="read", write_tool="write")
    context = _context()
    with pytest.raises(PolicyDeniedError, match="must follow"):
        policy.before_invoke(_tool("write"), context)
    policy.on_result(_tool("read"), ToolResult.error("denied"))
    with pytest.raises(PolicyDeniedError):
        policy.before_invoke(_tool("write"), context)
    policy.on_result(_tool("read"), ToolResult.ok("contents"))
    policy.before_invoke(_tool("write"), context)


def test_read_before_write_ignores_other_tools() -> None:
    policy = ReadBeforeWritePolicy(read_tool="r", write_tool="w")
    policy.before_invoke(_tool("other"), _context())  # should not raise


# ---------------------------------------------------------------------------
# MaxInvocationsPolicy
# ---------------------------------------------------------------------------


def test_max_invocations_for_tool_blocks_after_limit() -> None:
    policy = MaxInvocationsPolicy.for_tool("write", max_calls=2)
    policy.before_invoke(_tool("write"), _context())
    policy.before_invoke(_tool("write"), _context())
    with pytest.raises(PolicyDeniedError, match="max_calls=2"):
        policy.before_invoke(_tool("write"), _context())


def test_max_invocations_ignores_other_tools() -> None:
    policy = MaxInvocationsPolicy.for_tool("write", max_calls=1)
    policy.before_invoke(_tool("read"), _context())
    policy.before_invoke(_tool("read"), _context())
    policy.on_result(_tool("read"), ToolResult.ok(None))


def test_max_invocations_for_tools_shares_count() -> None:
    policy = MaxInvocationsPolicy.for_tools(("a", "b"), max_calls=2)
    policy.before_invoke(_tool("a"), _context())
    policy.before_invoke(_tool("b"), _context())
    with pytest.raises(PolicyDeniedError):
        policy.before_invoke(_tool("a"), _context())


def test_once_only_policy_allows_a_single_call_per_set() -> None:
    policy = OnceOnlyPolicy("apply")
    policy.before_invoke(_tool("apply"), _context())
    with pytest.raises(PolicyDeniedError):
        policy.before_invoke(_tool("apply"), _context())
