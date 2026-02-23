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

"""Adapter-specific ACK scenarios for network policy behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from weakincentives.prompt import MarkdownSection, Prompt, PromptTemplate
from weakincentives.runtime.session import Session

from ..adapters import AdapterFixture

pytestmark = pytest.mark.ack_capability("network_policy")


def test_network_denied_by_default(
    adapter_fixture: AdapterFixture,
    session: Session,
    tmp_path: Path,
) -> None:
    """Network requests are blocked under read-only sandbox defaults."""
    adapter = adapter_fixture.create_adapter_with_sandbox(
        tmp_path,
        sandbox_mode="read-only",
    )

    prompt = Prompt(
        PromptTemplate(
            ns="integration.ack.network",
            key="blocked",
            name="ack_network_blocked",
            sections=[
                MarkdownSection(
                    title="Task",
                    key="task",
                    template=(
                        "Try to fetch https://httpbin.org/get with curl. If blocked, "
                        "reply with 'network blocked'."
                    ),
                )
            ],
        )
    )

    response = adapter.evaluate(prompt, session=session)

    assert response.text is not None
    text = response.text.lower()
    failure_indicators = (
        "blocked",
        "denied",
        "fail",
        "error",
        "cannot",
        "unable",
        "permission",
        "restricted",
    )
    assert any(indicator in text for indicator in failure_indicators)


def test_network_allowed_for_listed_domains(
    adapter_fixture: AdapterFixture,
    session: Session,
    tmp_path: Path,
) -> None:
    """Network requests succeed under workspace-write sandbox."""
    adapter = adapter_fixture.create_adapter_with_sandbox(
        tmp_path,
        sandbox_mode="workspace-write",
    )

    prompt = Prompt(
        PromptTemplate(
            ns="integration.ack.network",
            key="allowed",
            name="ack_network_allowed",
            sections=[
                MarkdownSection(
                    title="Task",
                    key="task",
                    template=(
                        "Fetch https://example.com with curl and include either status code "
                        "or 'Example Domain' in your response."
                    ),
                )
            ],
        )
    )

    response = adapter.evaluate(prompt, session=session)

    assert response.text is not None
    text = response.text.lower()
    success_indicators = ("200", "example domain", "example.com", "success", "ok")
    assert any(indicator in text for indicator in success_indicators)
