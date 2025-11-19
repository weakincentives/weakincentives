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

"""Tests for the OpenAI adapter helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from types import SimpleNamespace
from typing import Any, cast

import pytest

from tests.helpers.events import NullEventBus
from weakincentives.adapters import openai as openai_module
from weakincentives.adapters.openai import (
    OpenAIAdapter,
    _extract_error_payload,
    _maybe_throttle_error,
    _retry_after_seconds,
    create_openai_client,
)
from weakincentives.prompt import Prompt
from weakincentives.prompt.prompt import RenderedPrompt
from weakincentives.prompt.structured_output import StructuredOutputConfig
from weakincentives.runtime.session import Session


class _ResponsePayload:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def json(self) -> dict[str, object]:
        return self._payload


class _RateLimitError(Exception):
    def __init__(self) -> None:
        super().__init__("rate limited")
        self.status_code = 429
        self.retry_after = "2.5"
        self.response = _ResponsePayload({"error": "too many requests"})


class _QuotaError(Exception):
    def __init__(self) -> None:
        super().__init__("quota exceeded")
        self.code = "insufficient_quota"
        self.retry_after = timedelta(seconds=4)
        self.response = None


class _MessageQuotaError(Exception):
    def __init__(self) -> None:
        super().__init__("request failed: insufficient_quota detected")
        self.response = None
        self.code = None
        self.retry_after = None


class APITimeoutError(Exception):
    pass


@dataclass(slots=True)
class _StructuredOutput:
    value: str


class _PromptStub:
    def __init__(self) -> None:
        self.ns = "tests"
        self.key = "prompt"
        self.name = "PromptStub"
        self.structured_output = StructuredOutputConfig(
            dataclass_type=_StructuredOutput,
            container="object",
            allow_extra_keys=False,
        )
        self.inject_output_instructions = True
        self.last_instructions: bool | None = None

    def render(
        self,
        *params: object,
        overrides_store: object | None = None,
        tag: str = "latest",
        inject_output_instructions: bool = True,
    ) -> RenderedPrompt[_StructuredOutput]:
        self.last_instructions = inject_output_instructions
        return RenderedPrompt(
            text="system",
            structured_output=self.structured_output,
        )


def test_maybe_throttle_error_handles_rate_limit_payload() -> None:
    error = _RateLimitError()

    throttle_error = _maybe_throttle_error(error)

    assert throttle_error is not None
    assert throttle_error.kind == "rate_limit"
    assert throttle_error.retry_after == pytest.approx(2.5)
    assert throttle_error.provider_payload == {"error": "too many requests"}


def test_maybe_throttle_error_handles_quota_signal_by_code() -> None:
    error = _QuotaError()

    throttle_error = _maybe_throttle_error(error)

    assert throttle_error is not None
    assert throttle_error.kind == "quota_exceeded"
    assert throttle_error.retry_after == pytest.approx(4.0)


def test_maybe_throttle_error_handles_quota_signal_in_message() -> None:
    error = _MessageQuotaError()

    throttle_error = _maybe_throttle_error(error)

    assert throttle_error is not None
    assert throttle_error.kind == "quota_exceeded"


def test_maybe_throttle_error_handles_timeout_error() -> None:
    error = APITimeoutError("timed out")

    throttle_error = _maybe_throttle_error(error)

    assert throttle_error is not None
    assert throttle_error.kind == "timeout"


def test_maybe_throttle_error_returns_none_for_other_errors() -> None:
    assert _maybe_throttle_error(RuntimeError("boom")) is None


def test_retry_after_seconds_handles_numeric_and_timedelta() -> None:
    assert _retry_after_seconds(2) == 2.0
    assert _retry_after_seconds(timedelta(seconds=3)) == 3.0


def test_extract_error_payload_uses_json_fallback() -> None:
    class Error(Exception):
        def __init__(self) -> None:
            self.response = SimpleNamespace(json=lambda: {"detail": "oops"})

    payload = _extract_error_payload(Error())

    assert payload == {"detail": "oops"}


def test_extract_error_payload_prefers_extract_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Response:
        pass

    class Error(Exception):
        def __init__(self) -> None:
            self.response = Response()

    monkeypatch.setattr(
        openai_module,
        "extract_payload",
        lambda response: {"detail": "boom"} if isinstance(response, Response) else None,
    )

    payload = _extract_error_payload(Error())

    assert payload == {"detail": "boom"}


def test_extract_error_payload_returns_none_without_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Response:
        pass

    class Error(Exception):
        def __init__(self) -> None:
            self.response = Response()

    monkeypatch.setattr(openai_module, "extract_payload", lambda response: None)

    assert _extract_error_payload(Error()) is None


def test_load_openai_module_raises_when_dependency_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _missing(_: str) -> object:
        raise ModuleNotFoundError

    monkeypatch.setattr(openai_module, "import_module", _missing)

    with pytest.raises(RuntimeError):
        openai_module._load_openai_module()


def test_load_openai_module_returns_module(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_module = SimpleNamespace(OpenAI=lambda **_: SimpleNamespace())

    monkeypatch.setattr(openai_module, "import_module", lambda _: stub_module)

    module = openai_module._load_openai_module()

    assert module is stub_module


def test_create_openai_client_uses_module_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[dict[str, object]] = []

    class _Client:
        def __init__(self, **kwargs: object) -> None:
            created.append(kwargs)

    stub_module = SimpleNamespace(OpenAI=_Client)
    monkeypatch.setattr(
        "weakincentives.adapters.openai._load_openai_module",
        lambda: stub_module,
    )

    client = create_openai_client(api_key="secret")

    assert isinstance(client, _Client)
    assert created == [{"api_key": "secret"}]


def test_openai_adapter_rejects_conflicting_client_factory() -> None:
    with pytest.raises(ValueError):
        OpenAIAdapter(
            model="test",
            client=SimpleNamespace(),
            client_factory=lambda **_: SimpleNamespace(),
        )


def test_openai_adapter_rejects_client_kwargs_with_explicit_client() -> None:
    with pytest.raises(ValueError):
        OpenAIAdapter(
            model="test",
            client=SimpleNamespace(),
            client_kwargs={"api_key": "secret"},
        )


def test_openai_adapter_uses_client_factory_when_client_missing() -> None:
    created: list[dict[str, object]] = []

    class _ClientStub:
        chat = SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_: SimpleNamespace(choices=[]))
        )

    def factory(**kwargs: object) -> object:
        created.append(kwargs)
        return _ClientStub()

    adapter = OpenAIAdapter(
        model="test",
        client_factory=cast(openai_module._OpenAIClientFactory, factory),
        client_kwargs={"api_key": "secret"},
    )

    assert isinstance(adapter, OpenAIAdapter)
    assert created == [{"api_key": "secret"}]


def test_openai_adapter_evaluate_disables_instructions_and_sets_response_format() -> (
    None
):
    captured_payloads: list[dict[str, Any]] = []

    class _ClientStub:
        def __init__(self) -> None:
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self._create)
            )

        def _create(self, **payload: object) -> object:
            captured_payloads.append(dict(payload))
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content="ok", tool_calls=[], parsed={"value": "ok"}
                        )
                    )
                ]
            )

    adapter = OpenAIAdapter(model="test", client=_ClientStub())
    prompt = _PromptStub()
    bus = NullEventBus()
    session = Session(bus=bus)

    response = adapter.evaluate(
        cast(Prompt[object], prompt),
        parse_output=True,
        bus=bus,
        session=session,
    )

    assert response.text is None
    assert response.output == _StructuredOutput(value="ok")
    assert prompt.last_instructions is False
    assert captured_payloads and "response_format" in captured_payloads[0]
