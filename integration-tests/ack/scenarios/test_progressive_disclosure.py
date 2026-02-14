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

"""Tier 3 ACK scenarios for progressive disclosure section expansion."""

from __future__ import annotations

import pytest

from weakincentives.adapters.core import ProviderAdapter
from weakincentives.prompt import Prompt, SectionVisibility, VisibilityExpansionRequired
from weakincentives.prompt.errors import SectionPath
from weakincentives.runtime.session import (
    Session,
    SetVisibilityOverride,
    VisibilityOverrides,
)

from ..adapters import AdapterFixture
from . import (
    InstructionParams,
    build_progressive_disclosure_prompt,
    build_verify_tool,
    make_adapter_ns,
)

pytestmark = pytest.mark.ack_capability("progressive_disclosure")

_MAX_EXPECTED_EXPANSIONS = 3


def test_two_level_hierarchy_expansion(
    adapter: ProviderAdapter[object],
    session: Session,
    adapter_fixture: AdapterFixture,
) -> None:
    """Adapters handle multi-step expansion and eventually reach leaf tools."""
    verify_tool = build_verify_tool()
    prompt = Prompt(
        build_progressive_disclosure_prompt(
            make_adapter_ns(adapter_fixture.adapter_name),
            verify_tool,
        )
    ).bind(InstructionParams(task="Verify integration-test-value"))

    expansion_count = 0
    tool_was_called = False

    while expansion_count < 5:
        try:
            response = adapter.evaluate(prompt, session=session)
            if response.text is not None:
                text_lower = response.text.lower()
                tool_was_called = (
                    "verified" in text_lower or "integration-test-value" in text_lower
                )
            break
        except VisibilityExpansionRequired as error:
            expansion_count += 1
            for path, visibility in error.requested_overrides.items():
                session.dispatch(
                    SetVisibilityOverride(path=path, visibility=visibility)
                )

    assert expansion_count > 0
    assert expansion_count <= _MAX_EXPECTED_EXPANSIONS
    assert tool_was_called

    overrides = session[VisibilityOverrides].latest()
    assert overrides is not None
    expected_paths: tuple[SectionPath, ...] = (
        ("instructions", "guidelines"),
        ("instructions", "guidelines", "tools-reference"),
    )
    for path in expected_paths:
        assert overrides.get(path) == SectionVisibility.FULL


def test_direct_leaf_expansion(
    adapter: ProviderAdapter[object],
    session: Session,
    adapter_fixture: AdapterFixture,
) -> None:
    """Adapters handle parent+leaf expansion requests issued in one step."""
    verify_tool = build_verify_tool()
    prompt = Prompt(
        build_progressive_disclosure_prompt(
            make_adapter_ns(adapter_fixture.adapter_name),
            verify_tool,
        )
    ).bind(
        InstructionParams(
            task=(
                "Expand all summarized sections and call verify_result with value "
                "integration-test-value"
            )
        )
    )

    final_response = None
    for _ in range(5):
        try:
            final_response = adapter.evaluate(prompt, session=session)
            break
        except VisibilityExpansionRequired as error:
            for path, visibility in error.requested_overrides.items():
                session.dispatch(
                    SetVisibilityOverride(path=path, visibility=visibility)
                )

    assert final_response is not None
    assert final_response.text is not None

    leaf_path: SectionPath = ("instructions", "guidelines", "tools-reference")
    overrides = session[VisibilityOverrides].latest()
    assert overrides is not None
    assert overrides.get(leaf_path) == SectionVisibility.FULL
