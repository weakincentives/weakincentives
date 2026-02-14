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

"""Tier 3 ACK scenarios for error handling semantics."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from weakincentives.adapters._shared._bridge import create_bridged_tools
from weakincentives.adapters.core import PromptEvaluationError, ProviderAdapter
from weakincentives.clock import FakeClock
from weakincentives.deadlines import Deadline
from weakincentives.prompt import Prompt
from weakincentives.runtime.session import Session

from ..adapters import AdapterFixture
from . import (
    GreetingParams,
    TransformRequest,
    build_greeting_prompt,
    build_tool_prompt,
    build_uppercase_tool,
    make_adapter_ns,
)


@pytest.mark.ack_capability("deadline_enforcement")
def test_deadline_enforcement(
    adapter: ProviderAdapter[object],
    session: Session,
    adapter_fixture: AdapterFixture,
) -> None:
    """Expired deadlines raise PromptEvaluationError before provider execution."""
    clock = FakeClock()
    deadline = Deadline(
        expires_at=clock.utcnow() + timedelta(seconds=2),
        clock=clock,
    )
    clock.advance(5)

    prompt = Prompt(
        build_greeting_prompt(make_adapter_ns(adapter_fixture.adapter_name))
    ).bind(GreetingParams(audience="expired deadline"))

    with pytest.raises(PromptEvaluationError) as exc_info:
        _ = adapter.evaluate(prompt, session=session, deadline=deadline)

    assert exc_info.value.phase in {"request", "budget"}


@pytest.mark.ack_capability("tool_invocation")
def test_invalid_tool_params_returns_error(
    session: Session,
    adapter_fixture: AdapterFixture,
    tmp_path: Path,
) -> None:
    """Invalid tool params return errors and do not poison subsequent calls."""
    calls: list[str] = []
    tool = build_uppercase_tool(calls)
    prompt = Prompt(
        build_tool_prompt(make_adapter_ns(adapter_fixture.adapter_name), tool)
    ).bind(TransformRequest(text="hello"))

    with prompt.resources:
        adapter = adapter_fixture.create_adapter(tmp_path)
        bridged = create_bridged_tools(
            (tool,),
            session=session,
            adapter=adapter,
            prompt=prompt,
            rendered_prompt=None,
            deadline=None,
            budget_tracker=None,
            adapter_name=adapter_fixture.adapter_name,
            prompt_name=prompt.name,
        )

        uppercase = bridged[0]
        invalid_result = uppercase({"text": 123})
        assert invalid_result.get("isError", False)

        valid_result = uppercase({"text": "hello"})
        assert not valid_result.get("isError", False)

    assert calls and calls[-1] == "hello"
