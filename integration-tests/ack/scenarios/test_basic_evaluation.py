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

"""Tier 1 ACK scenarios for basic prompt evaluation."""

from __future__ import annotations

import pytest

from weakincentives.adapters.core import ProviderAdapter
from weakincentives.prompt import Prompt
from weakincentives.runtime.session import Session

from ..adapters import AdapterFixture
from . import GreetingParams, build_greeting_prompt, make_adapter_ns

pytestmark = pytest.mark.ack_capability("text_response")


def test_returns_text_response(
    adapter: ProviderAdapter[object],
    session: Session,
    adapter_fixture: AdapterFixture,
) -> None:
    """ACK adapters return non-empty text for a simple greeting prompt."""
    prompt = Prompt(
        build_greeting_prompt(make_adapter_ns(adapter_fixture.adapter_name))
    ).bind(GreetingParams(audience="ack basic evaluation"))

    response = adapter.evaluate(prompt, session=session)

    assert response.text is not None
    assert response.text.strip()


def test_prompt_name_propagation(
    adapter: ProviderAdapter[object],
    session: Session,
    adapter_fixture: AdapterFixture,
) -> None:
    """ACK adapters preserve prompt template names in PromptResponse."""
    prompt = Prompt(
        build_greeting_prompt(make_adapter_ns(adapter_fixture.adapter_name))
    ).bind(GreetingParams(audience="ack prompt name"))

    response = adapter.evaluate(prompt, session=session)

    assert response.prompt_name == "ack_greeting"
