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

from datetime import UTC, datetime, timedelta
from typing import Any, cast

import pytest

from tests.adapters._test_stubs import DummyChoice, DummyMessage, DummyResponse
from tests.adapters.test_conversation_runner import RecordingBus, build_inner_loop
from weakincentives.adapters import litellm, openai
from weakincentives.adapters.core import PROMPT_EVALUATION_PHASE_REQUEST
from weakincentives.adapters.throttle import (
    ThrottleError,
    ThrottlePolicy,
    jittered_backoff,
    new_throttle_policy,
    sleep_for,
    throttle_details,
)
from weakincentives.deadlines import Deadline
from weakincentives.prompt.prompt import RenderedPrompt
from weakincentives.runtime.session import Session


def test_runner_retries_after_throttle(monkeypatch: pytest.MonkeyPatch) -> None:
    rendered = RenderedPrompt(text="system")
    dispatcher = RecordingBus()
    session = Session(dispatcher=dispatcher)
    response = DummyResponse([DummyChoice(DummyMessage(content="ok"))])
    delays: list[timedelta] = []
    calls = 0

    def _sleep(delay: timedelta) -> None:
        delays.append(delay)

    monkeypatch.setattr("weakincentives.adapters.inner_loop.sleep_for", _sleep)
    monkeypatch.setattr(
        "weakincentives.adapters.throttle.random.uniform", lambda _a, b: b
    )

    def provider(
        messages: list[dict[str, Any]],
        tool_specs: list[dict[str, Any]],
        tool_choice: object,
        response_format: object,
    ) -> DummyResponse:
        del messages, tool_specs, tool_choice, response_format
        nonlocal calls
        calls += 1
        if calls < 3:
            raise ThrottleError(
                "throttled",
                prompt_name="example",
                phase=PROMPT_EVALUATION_PHASE_REQUEST,
                details=throttle_details(
                    kind="rate_limit", retry_after=timedelta(seconds=1)
                ),
            )
        return response

    loop = build_inner_loop(
        rendered=rendered,
        provider=provider,  # type: ignore[arg-type]
        session=session,
        throttle_policy=new_throttle_policy(max_attempts=5),
    )

    result = loop.run()

    assert calls == 3
    assert delays[0] >= timedelta(seconds=1)
    assert result.text == "ok"


def test_runner_bubbles_throttle_when_budget_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rendered = RenderedPrompt(text="system")
    dispatcher = RecordingBus()
    session = Session(dispatcher=dispatcher)

    monkeypatch.setattr(
        "weakincentives.adapters.inner_loop.sleep_for", lambda _delay: None
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
            details=throttle_details(
                kind="rate_limit", retry_after=timedelta(milliseconds=50)
            ),
        )

    loop = build_inner_loop(
        rendered=rendered,
        provider=provider,  # type: ignore[arg-type]
        session=session,
        throttle_policy=new_throttle_policy(
            max_attempts=2, max_total_delay=timedelta(milliseconds=60)
        ),
    )

    with pytest.raises(ThrottleError) as excinfo:
        loop.run()

    error = cast(ThrottleError, excinfo.value)
    assert error.attempts == 1
    assert "budget" in error.message


def testjittered_backoff_returns_retry_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "weakincentives.adapters.throttle.random.uniform", lambda _a, _b: 0.0
    )

    policy = new_throttle_policy(base_delay=timedelta(milliseconds=10))
    retry_after = timedelta(milliseconds=30)

    delay = jittered_backoff(policy=policy, attempt=1, retry_after=retry_after)

    assert delay == retry_after


def test_throttle_policy_validation() -> None:
    with pytest.raises(ValueError):
        new_throttle_policy(max_attempts=0)
    with pytest.raises(ValueError):
        new_throttle_policy(base_delay=timedelta(0))
    with pytest.raises(ValueError):
        new_throttle_policy(max_delay=timedelta(0))
    with pytest.raises(ValueError):
        new_throttle_policy(max_total_delay=timedelta(0))


def test_runner_raises_on_non_retryable_throttle() -> None:
    rendered = RenderedPrompt(text="system")
    dispatcher = RecordingBus()
    session = Session(dispatcher=dispatcher)

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
            details=throttle_details(kind="rate_limit", retry_safe=False),
        )

    loop = build_inner_loop(
        rendered=rendered,
        provider=provider,  # type: ignore[arg-type]
        session=session,
        throttle_policy=new_throttle_policy(max_attempts=2),
    )

    with pytest.raises(ThrottleError):
        loop.run()


def test_runner_max_attempts_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    rendered = RenderedPrompt(text="system")
    dispatcher = RecordingBus()
    session = Session(dispatcher=dispatcher)
    monkeypatch.setattr(
        "weakincentives.adapters.inner_loop.sleep_for", lambda _delay: None
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
            details=throttle_details(
                kind="rate_limit", retry_after=timedelta(milliseconds=10)
            ),
        )

    loop = build_inner_loop(
        rendered=rendered,
        provider=provider,  # type: ignore[arg-type]
        session=session,
        throttle_policy=new_throttle_policy(
            max_attempts=1, base_delay=timedelta(milliseconds=10)
        ),
    )

    with pytest.raises(ThrottleError):
        loop.run()


def test_runner_deadline_prevents_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    rendered = RenderedPrompt(text="system")
    dispatcher = RecordingBus()
    session = Session(dispatcher=dispatcher)
    monkeypatch.setattr(
        "weakincentives.adapters.inner_loop.sleep_for", lambda _delay: None
    )
    monkeypatch.setattr(
        "weakincentives.adapters.inner_loop.jittered_backoff",
        lambda **_: timedelta(seconds=3),
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
            details=throttle_details(
                kind="rate_limit", retry_after=timedelta(seconds=1)
            ),
        )

    loop = build_inner_loop(
        rendered=rendered,
        provider=provider,  # type: ignore[arg-type]
        session=session,
        throttle_policy=new_throttle_policy(
            max_attempts=3, base_delay=timedelta(seconds=1)
        ),
        deadline=Deadline(expires_at=datetime.now(UTC) + timedelta(seconds=2)),
    )

    with pytest.raises(ThrottleError) as excinfo:
        loop.run()

    assert "Deadline expired" in str(excinfo.value)


def testjittered_backoff_respects_retry_after(monkeypatch: pytest.MonkeyPatch) -> None:
    policy = new_throttle_policy(
        base_delay=timedelta(seconds=1),
        max_delay=timedelta(seconds=4),
        max_total_delay=timedelta(seconds=10),
    )
    monkeypatch.setattr(
        "weakincentives.adapters.throttle.random.uniform", lambda _a, b: b
    )

    delay = jittered_backoff(policy=policy, attempt=2, retry_after=timedelta(seconds=2))

    assert delay == timedelta(seconds=2)


def testjittered_backoff_clamps_to_retry_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = new_throttle_policy(
        base_delay=timedelta(seconds=1),
        max_delay=timedelta(seconds=4),
        max_total_delay=timedelta(seconds=10),
    )
    monkeypatch.setattr(
        "weakincentives.adapters.throttle.random.uniform", lambda _a, _b: 0
    )

    delay = jittered_backoff(policy=policy, attempt=2, retry_after=timedelta(seconds=1))

    assert delay == timedelta(seconds=1)


def testjittered_backoff_with_non_positive_base() -> None:
    policy = ThrottlePolicy(
        max_attempts=1,
        base_delay=timedelta(0),
        max_delay=timedelta(seconds=1),
        max_total_delay=timedelta(seconds=1),
    )

    assert jittered_backoff(policy=policy, attempt=1, retry_after=None) == timedelta(0)


def test_sleep_for_invokes_time_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: list[float] = []
    monkeypatch.setattr("weakincentives.adapters.throttle.time.sleep", observed.append)

    sleep_for(timedelta(milliseconds=150))

    assert observed == [0.15]


class _ThrottleLikeError(Exception):
    def __init__(self, message: str, **attrs: object) -> None:
        super().__init__(message)
        for name, value in attrs.items():
            setattr(self, name, value)


def test_openai_retry_after_extraction_paths() -> None:
    direct = _ThrottleLikeError("rate limit", retry_after=2)
    header = _ThrottleLikeError("rate limit", headers={"Retry-After": "3"})
    response = _ThrottleLikeError(
        "rate limit",
        response={"retry_after": 4, "headers": {"retry-after": "5"}},
    )
    response_retry_after_only = _ThrottleLikeError(
        "rate limit", response={"retry_after": 6}
    )

    assert openai._retry_after_from_error(direct) == timedelta(seconds=2)
    assert openai._retry_after_from_error(header) == timedelta(seconds=3)
    assert openai._retry_after_from_error(response) == timedelta(seconds=5)
    assert openai._retry_after_from_error(response_retry_after_only) == timedelta(
        seconds=6
    )


def test_openai_throttle_normalization_and_payloads() -> None:
    error = _ThrottleLikeError(
        "insufficient_quota",
        code="insufficient_quota",
        response={"detail": "no quota"},
    )
    throttle = openai._normalize_openai_throttle(error, prompt_name="prompt")

    assert throttle is not None
    assert throttle.kind == "quota_exhausted"
    assert throttle.provider_payload == {"detail": "no quota"}

    timeout_error = _ThrottleLikeError("timeout", status_code=504)
    assert (
        openai._normalize_openai_throttle(timeout_error, prompt_name="prompt") is None
    )

    rate_error = _ThrottleLikeError("ratelimit", status_code=429)
    rate_throttle = openai._normalize_openai_throttle(rate_error, prompt_name="prompt")

    assert rate_throttle is not None
    assert rate_throttle.kind == "rate_limit"


def test_litellm_retry_after_and_normalization() -> None:
    header_error = _ThrottleLikeError("rate limited", headers={"retry-after": 1})
    response_error = _ThrottleLikeError(
        "rate limited",
        response={"headers": {"Retry-After": "2"}},
        status_code=429,
        code="rate_limit",
    )

    assert litellm._retry_after_from_error(header_error) == timedelta(seconds=1)
    assert litellm._retry_after_from_error(response_error) == timedelta(seconds=2)

    throttle = litellm._normalize_litellm_throttle(response_error, prompt_name="prompt")

    assert throttle is not None
    assert throttle.kind == "rate_limit"
    assert throttle.retry_after == timedelta(seconds=2)

    direct_error = _ThrottleLikeError("rate limited", retry_after=timedelta(seconds=3))
    assert litellm._retry_after_from_error(direct_error) == timedelta(seconds=3)

    response_retry_error = _ThrottleLikeError(
        "rate limited", response={"retry_after": "6"}
    )
    assert litellm._retry_after_from_error(response_retry_error) == timedelta(seconds=6)

    quota_error = _ThrottleLikeError("insufficient_quota", code="insufficient_quota")
    quota_throttle = litellm._normalize_litellm_throttle(
        quota_error, prompt_name="prompt"
    )

    assert quota_throttle is not None
    assert quota_throttle.kind == "quota_exhausted"

    timeout_error = type("TimeoutLiteError", (Exception,), {})("timeout")
    timeout_throttle = litellm._normalize_litellm_throttle(
        timeout_error, prompt_name="prompt"
    )

    assert timeout_throttle is not None
    assert timeout_throttle.kind == "timeout"

    payload_error = _ThrottleLikeError("rate limited", response={"detail": "x"})
    assert litellm._error_payload(payload_error) == {"detail": "x"}


def test_retry_after_coercion_variants() -> None:
    assert litellm._coerce_retry_after(timedelta(seconds=2)) == timedelta(seconds=2)
    assert litellm._coerce_retry_after(timedelta(seconds=-1)) is None
    assert litellm._coerce_retry_after("not-a-number") is None
    assert openai._coerce_retry_after(timedelta(seconds=4)) == timedelta(seconds=4)
    assert openai._coerce_retry_after(timedelta(seconds=-2)) is None
    assert openai._coerce_retry_after("7") == timedelta(seconds=7)
    assert openai._coerce_retry_after("abc") is None
    assert openai._coerce_retry_after(1.5) == timedelta(seconds=1.5)
    assert openai._coerce_retry_after(None) is None


def test_openai_additional_throttle_paths() -> None:
    timeout_class_error = type("TimeoutIssue", (Exception,), {})("timeout")
    timeout_throttle = openai._normalize_openai_throttle(
        timeout_class_error, prompt_name="prompt"
    )

    assert timeout_throttle is not None
    assert timeout_throttle.kind == "timeout"

    json_error = _ThrottleLikeError("other", json_body={"info": "payload"})
    assert openai._error_payload(json_error) == {"info": "payload"}

    neutral_error = _ThrottleLikeError("other")
    assert (
        litellm._normalize_litellm_throttle(neutral_error, prompt_name="prompt") is None
    )
