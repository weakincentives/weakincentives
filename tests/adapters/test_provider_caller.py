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

"""Tests for the ProviderCaller class."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import pytest

from tests.helpers import FrozenUtcNow, frozen_utcnow as _frozen_utcnow  # noqa: F401
from weakincentives.adapters.core import PROMPT_EVALUATION_PHASE_REQUEST
from weakincentives.adapters.provider_caller import ProviderCaller
from weakincentives.adapters.throttle import (
    ThrottleError,
    new_throttle_policy,
    throttle_details,
)
from weakincentives.deadlines import Deadline


@dataclass
class DummyResponse:
    """Simple response stub for testing."""

    content: str


def test_provider_caller_success() -> None:
    """Test that ProviderCaller returns response on success."""
    response = DummyResponse("Hello")

    def provider(
        messages: list[dict[str, Any]],
        tool_specs: Sequence[Mapping[str, Any]],
        tool_choice: object,
        response_format: Mapping[str, Any] | None,
    ) -> DummyResponse:
        del messages, tool_specs, tool_choice, response_format
        return response

    caller = ProviderCaller(
        call_provider=provider,
        prompt_name="test",
    )

    result = caller.call(
        [{"role": "user", "content": "Hi"}],
        [],
        None,
        None,
    )

    assert result is response


def test_provider_caller_passes_arguments() -> None:
    """Test that ProviderCaller passes all arguments to provider."""
    captured: dict[str, Any] = {}

    def provider(
        messages: list[dict[str, Any]],
        tool_specs: Sequence[Mapping[str, Any]],
        tool_choice: object,
        response_format: Mapping[str, Any] | None,
    ) -> DummyResponse:
        captured["messages"] = messages
        captured["tool_specs"] = tool_specs
        captured["tool_choice"] = tool_choice
        captured["response_format"] = response_format
        return DummyResponse("ok")

    caller = ProviderCaller(
        call_provider=provider,
        prompt_name="test",
    )

    messages = [{"role": "user", "content": "Hi"}]
    tool_specs = [{"type": "function", "name": "test"}]
    tool_choice = "auto"
    response_format = {"type": "json"}

    caller.call(messages, tool_specs, tool_choice, response_format)

    assert captured["messages"] == messages
    assert captured["tool_specs"] == tool_specs
    assert captured["tool_choice"] == tool_choice
    assert captured["response_format"] == response_format


def test_provider_caller_raises_on_expired_deadline(
    frozen_utcnow: FrozenUtcNow,
) -> None:
    """Test that ProviderCaller raises when deadline is already expired."""
    anchor = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    frozen_utcnow.set(anchor)
    deadline = Deadline(expires_at=anchor + timedelta(seconds=5))

    def provider(
        messages: list[dict[str, Any]],
        tool_specs: Sequence[Mapping[str, Any]],
        tool_choice: object,
        response_format: Mapping[str, Any] | None,
    ) -> DummyResponse:
        del messages, tool_specs, tool_choice, response_format
        return DummyResponse("should not reach")

    caller = ProviderCaller(
        call_provider=provider,
        prompt_name="test",
        deadline=deadline,
    )

    # Advance time past the deadline
    frozen_utcnow.advance(timedelta(seconds=10))

    from weakincentives.adapters.core import PromptEvaluationError

    with pytest.raises(PromptEvaluationError) as exc_info:
        caller.call([], [], None, None)

    assert exc_info.value.phase == PROMPT_EVALUATION_PHASE_REQUEST
    assert "Deadline expired" in str(exc_info.value)


def test_provider_caller_retries_on_throttle(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that ProviderCaller retries on ThrottleError."""
    calls = 0
    delays: list[timedelta] = []

    def _sleep(delay: timedelta) -> None:
        delays.append(delay)

    monkeypatch.setattr("weakincentives.adapters.provider_caller.sleep_for", _sleep)
    monkeypatch.setattr(
        "weakincentives.adapters.throttle.random.uniform", lambda _a, b: b
    )

    def provider(
        messages: list[dict[str, Any]],
        tool_specs: Sequence[Mapping[str, Any]],
        tool_choice: object,
        response_format: Mapping[str, Any] | None,
    ) -> DummyResponse:
        del messages, tool_specs, tool_choice, response_format
        nonlocal calls
        calls += 1
        if calls < 3:
            raise ThrottleError(
                "throttled",
                prompt_name="test",
                phase=PROMPT_EVALUATION_PHASE_REQUEST,
                details=throttle_details(
                    kind="rate_limit", retry_after=timedelta(seconds=1)
                ),
            )
        return DummyResponse("ok")

    caller = ProviderCaller(
        call_provider=provider,
        prompt_name="test",
        throttle_policy=new_throttle_policy(max_attempts=5),
    )

    result = caller.call([], [], None, None)

    assert calls == 3
    assert len(delays) == 2
    assert result.content == "ok"


def test_provider_caller_raises_on_non_retryable_throttle() -> None:
    """Test that ProviderCaller raises immediately on non-retryable throttle."""

    def provider(
        messages: list[dict[str, Any]],
        tool_specs: Sequence[Mapping[str, Any]],
        tool_choice: object,
        response_format: Mapping[str, Any] | None,
    ) -> DummyResponse:
        del messages, tool_specs, tool_choice, response_format
        raise ThrottleError(
            "throttled",
            prompt_name="test",
            phase=PROMPT_EVALUATION_PHASE_REQUEST,
            details=throttle_details(kind="rate_limit", retry_safe=False),
        )

    caller = ProviderCaller(
        call_provider=provider,
        prompt_name="test",
        throttle_policy=new_throttle_policy(max_attempts=5),
    )

    with pytest.raises(ThrottleError):
        caller.call([], [], None, None)


def test_provider_caller_raises_on_max_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that ProviderCaller raises when max attempts reached."""
    monkeypatch.setattr(
        "weakincentives.adapters.provider_caller.sleep_for", lambda _: None
    )

    def provider(
        messages: list[dict[str, Any]],
        tool_specs: Sequence[Mapping[str, Any]],
        tool_choice: object,
        response_format: Mapping[str, Any] | None,
    ) -> DummyResponse:
        del messages, tool_specs, tool_choice, response_format
        raise ThrottleError(
            "throttled",
            prompt_name="test",
            phase=PROMPT_EVALUATION_PHASE_REQUEST,
            details=throttle_details(
                kind="rate_limit", retry_after=timedelta(milliseconds=10)
            ),
        )

    caller = ProviderCaller(
        call_provider=provider,
        prompt_name="test",
        throttle_policy=new_throttle_policy(max_attempts=2),
    )

    with pytest.raises(ThrottleError) as exc_info:
        caller.call([], [], None, None)

    error = cast(ThrottleError, exc_info.value)
    assert "budget exhausted" in error.message


def test_provider_caller_raises_on_total_delay_exceeded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that ProviderCaller raises when total delay exceeded."""
    monkeypatch.setattr(
        "weakincentives.adapters.provider_caller.sleep_for", lambda _: None
    )

    def provider(
        messages: list[dict[str, Any]],
        tool_specs: Sequence[Mapping[str, Any]],
        tool_choice: object,
        response_format: Mapping[str, Any] | None,
    ) -> DummyResponse:
        del messages, tool_specs, tool_choice, response_format
        raise ThrottleError(
            "throttled",
            prompt_name="test",
            phase=PROMPT_EVALUATION_PHASE_REQUEST,
            details=throttle_details(
                kind="rate_limit", retry_after=timedelta(milliseconds=50)
            ),
        )

    caller = ProviderCaller(
        call_provider=provider,
        prompt_name="test",
        throttle_policy=new_throttle_policy(
            max_attempts=10, max_total_delay=timedelta(milliseconds=60)
        ),
    )

    with pytest.raises(ThrottleError) as exc_info:
        caller.call([], [], None, None)

    error = cast(ThrottleError, exc_info.value)
    assert "window exceeded" in error.message


def test_provider_caller_raises_on_deadline_during_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that ProviderCaller raises when deadline expires during retry."""
    monkeypatch.setattr(
        "weakincentives.adapters.provider_caller.sleep_for", lambda _: None
    )
    monkeypatch.setattr(
        "weakincentives.adapters.provider_caller.jittered_backoff",
        lambda **_: timedelta(seconds=5),
    )

    deadline = Deadline(expires_at=datetime.now(UTC) + timedelta(seconds=2))

    def provider(
        messages: list[dict[str, Any]],
        tool_specs: Sequence[Mapping[str, Any]],
        tool_choice: object,
        response_format: Mapping[str, Any] | None,
    ) -> DummyResponse:
        del messages, tool_specs, tool_choice, response_format
        raise ThrottleError(
            "throttled",
            prompt_name="test",
            phase=PROMPT_EVALUATION_PHASE_REQUEST,
            details=throttle_details(
                kind="rate_limit", retry_after=timedelta(seconds=1)
            ),
        )

    caller = ProviderCaller(
        call_provider=provider,
        prompt_name="test",
        deadline=deadline,
        throttle_policy=new_throttle_policy(max_attempts=5),
    )

    with pytest.raises(ThrottleError) as exc_info:
        caller.call([], [], None, None)

    assert "Deadline expired" in str(exc_info.value)


def test_provider_caller_no_deadline_succeeds() -> None:
    """Test that ProviderCaller works without a deadline."""

    def provider(
        messages: list[dict[str, Any]],
        tool_specs: Sequence[Mapping[str, Any]],
        tool_choice: object,
        response_format: Mapping[str, Any] | None,
    ) -> DummyResponse:
        del messages, tool_specs, tool_choice, response_format
        return DummyResponse("ok")

    caller = ProviderCaller(
        call_provider=provider,
        prompt_name="test",
        deadline=None,  # Explicitly no deadline
    )

    result = caller.call([], [], None, None)
    assert result.content == "ok"


def test_provider_caller_respects_error_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that ProviderCaller uses attempts from ThrottleError."""
    monkeypatch.setattr(
        "weakincentives.adapters.provider_caller.sleep_for", lambda _: None
    )
    calls = 0

    def provider(
        messages: list[dict[str, Any]],
        tool_specs: Sequence[Mapping[str, Any]],
        tool_choice: object,
        response_format: Mapping[str, Any] | None,
    ) -> DummyResponse:
        del messages, tool_specs, tool_choice, response_format
        nonlocal calls
        calls += 1
        # Error claims 3 attempts already happened
        raise ThrottleError(
            "throttled",
            prompt_name="test",
            phase=PROMPT_EVALUATION_PHASE_REQUEST,
            details=throttle_details(
                kind="rate_limit",
                attempts=3,
                retry_after=timedelta(milliseconds=10),
            ),
        )

    # Only allow 3 attempts total - should fail on first retry
    caller = ProviderCaller(
        call_provider=provider,
        prompt_name="test",
        throttle_policy=new_throttle_policy(max_attempts=3),
    )

    with pytest.raises(ThrottleError) as exc_info:
        caller.call([], [], None, None)

    error = cast(ThrottleError, exc_info.value)
    assert calls == 1  # Only one actual call - error claimed 3 attempts
    assert error.attempts == 3
