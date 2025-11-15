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

"""Regression tests for the interactive code reviewer example."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pytest import CaptureFixture

from code_reviewer_example import build_code_reviewer_state
from tests.helpers.adapters import UNIT_TEST_ADAPTER_NAME
from weakincentives.prompt.overrides import LocalPromptOverridesStore
from weakincentives.runtime.events import PromptRendered


def test_prompt_render_reducer_prints_full_prompt(
    capsys: CaptureFixture[str],
    tmp_path: Path,
) -> None:
    overrides_store = LocalPromptOverridesStore(root_path=tmp_path)
    _prompt, session, bus = build_code_reviewer_state(
        overrides_store=overrides_store,
        override_tag="prompt-log",
    )

    event = PromptRendered(
        prompt_ns="example",
        prompt_key="sunfish",
        prompt_name="sunfish_code_review_agent",
        adapter=UNIT_TEST_ADAPTER_NAME,
        session_id=session.session_id,
        render_inputs=(),
        rendered_prompt="<prompt body>",
        created_at=datetime.now(UTC),
        event_id=uuid4(),
    )

    publish_result = bus.publish(event)
    assert publish_result.handled_count >= 1

    captured = capsys.readouterr()
    assert "Rendered prompt" in captured.out
    assert "<prompt body>" in captured.out

    stored_events = session.select_all(PromptRendered)
    assert stored_events
    assert stored_events[-1].rendered_prompt == "<prompt body>"
