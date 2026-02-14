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

"""Adapter-specific ACK scenario for native tool event emission."""

from __future__ import annotations

from pathlib import Path

import pytest

from weakincentives.prompt import Prompt
from weakincentives.runtime.events.types import ToolInvoked
from weakincentives.runtime.session import Session

from ..adapters import AdapterFixture
from . import build_native_tool_prompt, make_adapter_ns

pytestmark = pytest.mark.ack_capability("native_tools")


def test_native_tool_emits_tool_invoked(
    adapter_fixture: AdapterFixture,
    session: Session,
    tmp_path: Path,
) -> None:
    """Native provider tools are reported as ToolInvoked telemetry events."""
    (tmp_path / "README.md").write_text(
        "# ACK Native Tool\n\nThis file is used by ACK.\n"
    )

    adapter = adapter_fixture.create_adapter(tmp_path)
    prompt = Prompt(
        build_native_tool_prompt(make_adapter_ns(adapter_fixture.adapter_name))
    )

    tool_events: list[ToolInvoked] = []
    session.dispatcher.subscribe(ToolInvoked, tool_events.append)

    response = adapter.evaluate(prompt, session=session)

    assert response.text is not None
    assert tool_events
    assert all(event.adapter == adapter_fixture.adapter_name for event in tool_events)
