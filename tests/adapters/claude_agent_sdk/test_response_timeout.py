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

"""Tests for deadline-exceeded path in resolve_response_wait_timeout."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from datetime import timedelta
from typing import Any
from unittest.mock import MagicMock

from weakincentives.deadlines import Deadline
from weakincentives.prompt import Prompt, PromptTemplate
from weakincentives.runtime.session import Session


class TestResolveResponseWaitTimeout:
    """Tests for deadline-exceeded path in resolve_response_wait_timeout."""

    def _make_expired_hook_context(self, session: Session) -> Any:
        from weakincentives.adapters.claude_agent_sdk._hooks import (
            HookConstraints,
            HookContext,
        )

        mock_deadline = MagicMock(spec=Deadline)
        mock_deadline.remaining.return_value = timedelta(seconds=-1)

        template: PromptTemplate[None] = PromptTemplate(
            ns="test", key="test", name="test"
        )
        prompt: Prompt[None] = Prompt(template)
        constraints = HookConstraints(deadline=mock_deadline)
        return HookContext(
            prompt=prompt,
            session=session,
            adapter_name="test",
            prompt_name="test",
            constraints=constraints,
        )

    def test_deadline_already_expired_returns_stop(self, session: Session) -> None:
        """When deadline remaining <= 0, should signal stop."""
        from weakincentives.adapters.claude_agent_sdk._sdk_execution import (
            resolve_response_wait_timeout,
        )

        hook_context = self._make_expired_hook_context(session)

        timeout_val, should_stop = resolve_response_wait_timeout(
            hook_context=hook_context,
            continuation_round=0,
            message_count=0,
        )
        assert should_stop is True
        assert timeout_val is None

    def test_next_response_message_returns_none_on_stop(self, session: Session) -> None:
        """next_response_message returns None when deadline expired."""
        from weakincentives.adapters.claude_agent_sdk._sdk_execution import (
            next_response_message,
        )

        async def _run() -> None:
            hook_context = self._make_expired_hook_context(session)

            async def fake_stream() -> AsyncGenerator[object, None]:
                yield "should not reach"

            result = await next_response_message(
                response_stream=fake_stream().__aiter__(),
                hook_context=hook_context,
                continuation_round=0,
                message_count=0,
            )
            assert result is None

        asyncio.run(_run())
