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

"""Tests for the LiteLLM adapter helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from types import SimpleNamespace
from typing import Any, cast

import pytest

from tests.helpers.events import NullEventBus
from weakincentives.adapters import litellm as litellm_module
from weakincentives.adapters.litellm import (
    LiteLLMAdapter,
    LiteLLMCompletion,
    _maybe_throttle_error,
    create_litellm_completion,
)
from weakincentives.prompt import Prompt
from weakincentives.prompt.prompt import RenderedPrompt
from weakincentives.prompt.structured_output import StructuredOutputConfig
from weakincentives.runtime.session import Session


class _RateLimitError(Exception):
    def __init__(self) -> None:
        super().__init__("rate limited")
        self.status_code = 429
        self.retry_after = "1.25"


class _QuotaError(Exception):
    def __init__(self) -> None:
        super().__init__("quota exceeded")
        self.error_code = "insufficient_quota"
        self.retry_after = timedelta(seconds=3)


class _OtherError(Exception):
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


_COMPLETION_RESPONSE = SimpleNamespace(
    choices=[
        SimpleNamespace(
            message=SimpleNamespace(content="ok", tool_calls=[], parsed={"value": "ok"})
        )
    ]
)


def _completion_callable(*args: object, **kwargs: object) -> object:
    return _COMPLETION_RESPONSE


COMPLETION_STUB = cast(LiteLLMCompletion, _completion_callable)


def test_maybe_throttle_error_handles_rate_limit_payload() -> None:
    error = _RateLimitError()

    throttle_error = _maybe_throttle_error(error)

    assert throttle_error is not None
    assert throttle_error.kind == "rate_limit"
    assert throttle_error.retry_after == pytest.approx(1.25)


def test_maybe_throttle_error_handles_quota_signal() -> None:
    error = _QuotaError()

    throttle_error = _maybe_throttle_error(error)

    assert throttle_error is not None
    assert throttle_error.kind == "quota_exceeded"
    assert throttle_error.retry_after == pytest.approx(3.0)


def test_maybe_throttle_error_returns_none_for_non_throttle_error() -> None:
    assert _maybe_throttle_error(_OtherError("boom")) is None


def test_load_litellm_module_raises_when_dependency_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _missing(_: str) -> object:
        raise ModuleNotFoundError

    monkeypatch.setattr(litellm_module, "import_module", _missing)

    with pytest.raises(RuntimeError):
        litellm_module._load_litellm_module()


def test_load_litellm_module_returns_module(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_module = SimpleNamespace(completion=lambda **_: None)

    monkeypatch.setattr(litellm_module, "import_module", lambda _: stub_module)

    module = litellm_module._load_litellm_module()

    assert module is stub_module


def test_create_litellm_completion_returns_module_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_module = SimpleNamespace(completion=lambda **kwargs: kwargs)
    monkeypatch.setattr(
        "weakincentives.adapters.litellm._load_litellm_module",
        lambda: stub_module,
    )

    completion = create_litellm_completion()

    assert completion is stub_module.completion


def test_create_litellm_completion_merges_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: list[dict[str, object]] = []

    def _completion(*args: object, **payload: object) -> dict[str, object]:
        recorded.append(dict(payload))
        return dict(payload)

    stub_module = SimpleNamespace(completion=_completion)
    monkeypatch.setattr(
        "weakincentives.adapters.litellm._load_litellm_module",
        lambda: stub_module,
    )

    completion = create_litellm_completion(timeout=5)
    completion(model="test", messages=[])

    assert recorded and recorded[0]["timeout"] == 5


def test_litellm_adapter_rejects_conflicting_completion_factory() -> None:
    with pytest.raises(ValueError):
        LiteLLMAdapter(
            model="test",
            completion=COMPLETION_STUB,
            completion_factory=lambda **_: COMPLETION_STUB,
        )


def test_litellm_adapter_rejects_completion_kwargs_with_explicit_completion() -> None:
    with pytest.raises(ValueError):
        LiteLLMAdapter(
            model="test",
            completion=COMPLETION_STUB,
            completion_kwargs={"timeout": 5},
        )


def test_litellm_adapter_uses_completion_factory_when_needed() -> None:
    created: list[dict[str, object]] = []

    def factory(**kwargs: object) -> LiteLLMCompletion:
        created.append(kwargs)

        def _complete(*args: object, **payload: object) -> object:
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content="ok", tool_calls=[], parsed={"value": "ok"}
                        )
                    )
                ]
            )

        return cast(LiteLLMCompletion, _complete)

    adapter = LiteLLMAdapter(
        model="test",
        completion_factory=factory,
        completion_kwargs={"timeout": 5},
    )

    prompt = _PromptStub()
    bus = NullEventBus()
    session = Session(bus=bus)
    adapter.evaluate(
        cast(Prompt[object], prompt), parse_output=False, bus=bus, session=session
    )

    assert created == [{"timeout": 5}]


def test_litellm_adapter_evaluate_disables_instructions_and_sets_response_format() -> (
    None
):
    captured_payloads: list[dict[str, Any]] = []

    def completion(*args: object, **payload: object) -> object:
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

    adapter = LiteLLMAdapter(
        model="test", completion=cast(LiteLLMCompletion, completion)
    )
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
