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

"""Tests for the spine extension protocols."""

from __future__ import annotations

from datetime import timedelta

from weakincentives.core.protocols import (
    Budget,
    Deadline,
    EventListener,
    PromptResponse,
    ProviderAdapter,
    ResourceProvider,
    ToolCall,
    Usage,
)


class _FakeDeadline:
    def remaining(self) -> timedelta:
        return timedelta(seconds=10)

    def expired(self) -> bool:
        return False


class _FakeBudget:
    def __init__(self) -> None:
        self._consumed = Usage()

    def record(self, usage: Usage) -> None:
        self._consumed = usage

    def check(self) -> None:
        return

    @property
    def consumed(self) -> Usage:
        return self._consumed


class _FakeListener:
    def __init__(self) -> None:
        self.events: list[object] = []

    def on_event(self, event: object) -> None:
        self.events.append(event)


class _FakeProvider:
    def get[T](self, protocol_type: type[T]) -> T:
        del protocol_type
        raise LookupError("not provided")


class _FakeAdapter:
    def evaluate(
        self,
        rendered: object,
        session: object,
        *,
        deadline: Deadline | None = None,
    ) -> PromptResponse:
        del rendered, session, deadline
        return PromptResponse(text="hi")


def test_usage_defaults() -> None:
    usage = Usage()
    assert usage.input_tokens == 0
    assert usage.output_tokens == 0
    assert usage.total_tokens == 0


def test_tool_call_defaults() -> None:
    call = ToolCall(name="t")
    assert call.arguments == {}
    assert call.call_id is None


def test_prompt_response_defaults() -> None:
    response = PromptResponse(text="hi")
    assert response.tool_calls == ()
    assert response.usage.total_tokens == 0
    assert response.finish_reason is None


def test_protocols_are_runtime_checkable() -> None:
    assert isinstance(_FakeDeadline(), Deadline)
    assert isinstance(_FakeBudget(), Budget)
    assert isinstance(_FakeListener(), EventListener)
    assert isinstance(_FakeProvider(), ResourceProvider)
    assert isinstance(_FakeAdapter(), ProviderAdapter)
