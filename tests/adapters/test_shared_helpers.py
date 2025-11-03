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

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

import pytest

from weakincentives.adapters import PromptEvaluationError, shared


def test_first_choice_returns_first_item() -> None:
    response = SimpleNamespace(choices=["first", "second"])

    assert shared.first_choice(response, prompt_name="example") == "first"


def test_first_choice_requires_sequence() -> None:
    response = SimpleNamespace(choices=None)

    with pytest.raises(PromptEvaluationError):
        shared.first_choice(response, prompt_name="example")


def test_parse_tool_arguments_rejects_non_string_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_loads(_: str) -> Mapping[Any, Any]:
        # Simulate a mapping that does not use string keys to exercise defensive branch.
        return {1: "value"}

    monkeypatch.setattr(shared.json, "loads", fake_loads)

    with pytest.raises(PromptEvaluationError) as err:
        shared.parse_tool_arguments(
            "{}",
            prompt_name="example",
            provider_payload=None,
        )

    message = str(err.value)
    assert "string keys" in message


def test_mapping_to_str_dict_rejects_non_string_keys() -> None:
    assert shared._mapping_to_str_dict({1: "value"}) is None
