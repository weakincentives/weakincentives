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
from typing import Any, cast
from uuid import uuid4

from pytest import CaptureFixture

from code_reviewer_example import (
    CodeReviewApp,
    RepositoryOptimizationResponse,
    ReviewResponse,
    ReviewTurnParams,
    build_task_prompt,
    initialize_code_reviewer_runtime,
    save_repository_instructions_override,
)
from tests.helpers.adapters import UNIT_TEST_ADAPTER_NAME
from weakincentives.adapters import PromptResponse
from weakincentives.adapters.core import ProviderAdapter
from weakincentives.prompt import Prompt, SupportsDataclass
from weakincentives.prompt.overrides import (
    LocalPromptOverridesStore,
    PromptDescriptor,
)
from weakincentives.runtime.events import InProcessEventBus, PromptRendered
from weakincentives.runtime.session import Session


class _RepositoryOptimizationAdapter:
    """Stub adapter that emits canned repository instructions."""

    def __init__(self, instructions: str) -> None:
        self.instructions = instructions
        self.calls: list[str] = []

    def evaluate(
        self,
        prompt: Prompt[SupportsDataclass],
        *params: SupportsDataclass,
        parse_output: bool = True,
        bus: InProcessEventBus | None = None,
        session: Session | None = None,
        deadline: object | None = None,
        overrides_store: LocalPromptOverridesStore | None = None,
        overrides_tag: str = "latest",
    ) -> PromptResponse[Any]:
        del params, parse_output, bus, session, deadline, overrides_store, overrides_tag
        self.calls.append(prompt.key)
        if prompt.key == "sunfish-repository-optimize":
            return PromptResponse(
                prompt_name=prompt.name or prompt.key,
                text=None,
                output=RepositoryOptimizationResponse(instructions=self.instructions),
                tool_results=(),
                provider_payload=None,
            )
        return PromptResponse(
            prompt_name=prompt.name or prompt.key,
            text="",
            output=None,
            tool_results=(),
            provider_payload=None,
        )


def test_prompt_render_reducer_prints_full_prompt(
    capsys: CaptureFixture[str],
    tmp_path: Path,
) -> None:
    overrides_store = LocalPromptOverridesStore(root_path=tmp_path)
    (
        _prompt,
        session,
        bus,
        _store,
        _tag,
    ) = initialize_code_reviewer_runtime(
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


def test_repository_instructions_section_empty_by_default() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    prompt = build_task_prompt(session=session)

    rendered = prompt.render(ReviewTurnParams(request="demo request"))

    assert "## Repository Instructions" in rendered.text
    post_section = rendered.text.split("## Repository Instructions", 1)[1]
    section_body = post_section.split("\n## ", 1)[0]
    assert section_body.strip() == ""


def test_save_repository_instructions_override_writes_body(tmp_path: Path) -> None:
    overrides_store = LocalPromptOverridesStore(root_path=tmp_path)
    bus = InProcessEventBus()
    session = Session(bus=bus)
    prompt = build_task_prompt(session=session)

    body = "- Custom instructions"
    save_repository_instructions_override(
        prompt=prompt,
        overrides_store=overrides_store,
        overrides_tag="seed",
        body=body,
    )

    descriptor = PromptDescriptor.from_prompt(prompt)
    override = overrides_store.resolve(descriptor=descriptor, tag="seed")
    assert override is not None
    assert override.sections["repository-instructions",].body == body


def test_repository_instructions_override_escapes_dollar_signs(tmp_path: Path) -> None:
    overrides_store = LocalPromptOverridesStore(root_path=tmp_path)
    bus = InProcessEventBus()
    session = Session(bus=bus)
    prompt = build_task_prompt(session=session)

    body = "- Define $PATH and ${VAR}"
    save_repository_instructions_override(
        prompt=prompt,
        overrides_store=overrides_store,
        overrides_tag="seed",
        body=body,
    )

    descriptor = PromptDescriptor.from_prompt(prompt)
    override = overrides_store.resolve(descriptor=descriptor, tag="seed")
    assert override is not None
    stored_body = override.sections["repository-instructions",].body
    assert "$$PATH" in stored_body
    assert "$${VAR}" in stored_body

    rendered = prompt.render(
        ReviewTurnParams(request="demo request"),
        overrides_store=overrides_store,
        tag="seed",
    )
    assert "$PATH" in rendered.text
    assert "${VAR}" in rendered.text


def test_optimize_command_persists_override(tmp_path: Path) -> None:
    overrides_store = LocalPromptOverridesStore(root_path=tmp_path)
    adapter = _RepositoryOptimizationAdapter("- Repo instructions from stub")
    app = CodeReviewApp(
        cast(ProviderAdapter[ReviewResponse], adapter),
        overrides_store=overrides_store,
    )

    assert app.override_tag == str(app.session.session_id)

    app._handle_optimize_command("Highlight docs")

    descriptor = PromptDescriptor.from_prompt(app.prompt)
    override = overrides_store.resolve(descriptor=descriptor, tag=app.override_tag)
    assert override is not None
    saved_body = override.sections["repository-instructions",].body
    assert saved_body == "- Repo instructions from stub"
    assert adapter.calls == ["sunfish-repository-optimize"]
