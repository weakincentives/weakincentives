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

"""Tests for stop, user prompt submit, subagent stop, and pre-compact hooks."""

from __future__ import annotations

import asyncio

from weakincentives.adapters.claude_agent_sdk._hooks import (
    HookContext,
    create_pre_compact_hook,
    create_stop_hook,
    create_subagent_stop_hook,
    create_user_prompt_submit_hook,
)


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
