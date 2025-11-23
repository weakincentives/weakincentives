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

from datetime import timedelta
from typing import Any

import pytest

from tests.adapters._test_stubs import DummyChoice, DummyMessage, DummyResponse
from tests.adapters.test_conversation_runner import RecordingBus, build_runner
from weakincentives.adapters.core import PROMPT_EVALUATION_PHASE_REQUEST
from weakincentives.adapters.shared import ThrottleError, new_throttle_policy
from weakincentives.prompt.prompt import RenderedPrompt


def test_runner_retries_after_throttle(monkeypatch: pytest.MonkeyPatch) -> None:
    rendered = RenderedPrompt(text="system")
    bus = RecordingBus()
    response = DummyResponse([DummyChoice(DummyMessage(content="ok"))])
    delays: list[timedelta] = []

    def _sleep(delay: timedelta) -> None:
        delays.append(delay)

    monkeypatch.setattr("weakincentives.adapters.shared._sleep_for", _sleep)
    monkeypatch.setattr(
        "weakincentives.adapters.shared.random.uniform", lambda _a, b: b
    )

    def provider(
        messages: list[dict[str, Any]],
        tool_specs: list[dict[str, Any]],
        tool_choice: object,
        response_format: object,
    ) -> DummyResponse:
        del messages, tool_specs, tool_choice, response_format
        provider.calls += 1
        if provider.calls < 3:
            raise ThrottleError(
                "throttled",
                prompt_name="example",
                phase=PROMPT_EVALUATION_PHASE_REQUEST,
                kind="rate_limit",
                retry_after=timedelta(seconds=1),
            )
        return response

    provider.calls = 0  # type: ignore[attr-defined]

    runner = build_runner(
        rendered=rendered,
        provider=provider,  # type: ignore[arg-type]
        bus=bus,
        throttle_policy=new_throttle_policy(max_attempts=5),
    )

    result = runner.run()

    assert provider.calls == 3
    assert delays[0] >= timedelta(seconds=1)
    assert result.text == "ok"


def test_runner_bubbles_throttle_when_budget_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rendered = RenderedPrompt(text="system")
    bus = RecordingBus()

    monkeypatch.setattr(
        "weakincentives.adapters.shared._sleep_for", lambda _delay: None
    )

    def provider(
        messages: list[dict[str, Any]],
        tool_specs: list[dict[str, Any]],
        tool_choice: object,
        response_format: object,
    ) -> DummyResponse:
        del messages, tool_specs, tool_choice, response_format
        raise ThrottleError(
            "throttled",
            prompt_name="example",
            phase=PROMPT_EVALUATION_PHASE_REQUEST,
            kind="rate_limit",
            retry_after=timedelta(milliseconds=50),
        )

    runner = build_runner(
        rendered=rendered,
        provider=provider,  # type: ignore[arg-type]
        bus=bus,
        throttle_policy=new_throttle_policy(
            max_attempts=2, max_total_delay=timedelta(milliseconds=60)
        ),
    )

    with pytest.raises(ThrottleError) as excinfo:
        runner.run()

    error = excinfo.value
    assert error.attempts == 1
    assert "budget" in error.message
