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

"""Interactive example that mounts the sunfish repo for agent exploration."""

from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path
from typing import Any

from weakincentives.adapters.litellm import LiteLLMAdapter
from weakincentives.adapters.openai import OpenAIAdapter
from weakincentives.events import InProcessEventBus, ToolInvoked
from weakincentives.examples.code_review_prompt import (
    ReviewGuidance,
    ReviewResponse,
    ReviewTurnParams,
)
from weakincentives.examples.code_review_session import (
    SupportsReviewEvaluate,
    ToolCallLog,
)
from weakincentives.prompt import MarkdownSection, Prompt
from weakincentives.serde import dump
from weakincentives.session import Session, select_latest
from weakincentives.tools.planning import Plan, PlanningToolsSection
from weakincentives.tools.vfs import HostMount, VfsPath, VfsToolsSection

PROJECT_ROOT = Path(__file__).resolve().parent
TEST_REPOSITORIES_ROOT = (PROJECT_ROOT / "test-repositories").resolve()
SUNFISH_MOUNT_INCLUDE_GLOBS: tuple[str, ...] = (
    "*.md",
    "*.py",
    "*.txt",
    "*.yml",
    "*.yaml",
    "*.toml",
    "*.gitignore",
    "*.json",
    "*.cfg",
    "*.ini",
    "*.sh",
    "*.6",
)
SUNFISH_MOUNT_EXCLUDE_GLOBS: tuple[str, ...] = (
    "**/*.pickle",
    "**/*.png",
    "**/*.bmp",
    "README.md",
)
SUNFISH_MOUNT_MAX_BYTES = 600_000


def build_sunfish_prompt(session: Session) -> Prompt[ReviewResponse]:
    if not TEST_REPOSITORIES_ROOT.exists():
        raise SystemExit(
            f"Expected test repositories under {TEST_REPOSITORIES_ROOT!s},"
            " but the directory is missing."
        )

    guidance_section = MarkdownSection[ReviewGuidance](
        title="Code Review Brief",
        template=textwrap.dedent(
            """
            You are a code review assistant exploring the pre-mounted virtual
            filesystem. The sunfish sample repository is available inside the
            VFS under the `sunfish/` directory.

            Use the available tools to stay grounded:
            - Planning tools help you capture multi-step investigations; keep the
              plan updated as you explore.
            - VFS tools list directories, read files, and stage edits. Mounted
              files are read-only; use writes to stage new snapshots.

            Respond with JSON containing:
            - summary: One paragraph describing your findings so far.
            - issues: List concrete risks, questions, or follow-ups you found.
            - next_steps: Actionable recommendations to progress the task.
            """
        ).strip(),
        default_params=ReviewGuidance(),
        key="code-review-brief",
    )
    planning_section = PlanningToolsSection(session=session)
    vfs_section = VfsToolsSection(
        session=session,
        mounts=(
            HostMount(
                host_path="sunfish",
                mount_path=VfsPath(("sunfish",)),
                include_glob=SUNFISH_MOUNT_INCLUDE_GLOBS,
                exclude_glob=SUNFISH_MOUNT_EXCLUDE_GLOBS,
                max_bytes=SUNFISH_MOUNT_MAX_BYTES,
            ),
        ),
        allowed_host_roots=(TEST_REPOSITORIES_ROOT,),
    )
    user_turn_section = MarkdownSection[ReviewTurnParams](
        title="Review Request",
        template="${request}",
        key="review-request",
    )
    return Prompt[ReviewResponse](
        ns="examples/code-review",
        key="code-review-session",
        name="sunfish_code_review_agent",
        sections=(guidance_section, planning_section, vfs_section, user_turn_section),
    )


class SunfishReviewSession:
    """Interactive session that records tool calls and plan snapshots."""

    def __init__(self, adapter: SupportsReviewEvaluate) -> None:
        self._adapter = adapter
        self._bus = InProcessEventBus()
        self._session = Session(bus=self._bus)
        self._prompt = build_sunfish_prompt(self._session)
        self._history: list[ToolCallLog] = []
        self._bus.subscribe(ToolInvoked, self._on_tool_invoked)

    def evaluate(self, request: str) -> str:
        response = self._adapter.evaluate(
            self._prompt,
            ReviewTurnParams(request=request),
            bus=self._bus,
        )
        if response.output is not None:
            rendered_output = dump(response.output, exclude_none=True)
            return json.dumps(rendered_output, ensure_ascii=False, indent=2)
        if response.text:
            return response.text
        return "(no response from assistant)"

    def render_tool_history(self) -> str:
        if not self._history:
            return "No tool calls recorded yet."

        lines: list[str] = []
        for index, record in enumerate(self._history, start=1):
            lines.append(
                f"{index}. {record.name} ({record.prompt_name}) → {record.message}"
            )
            if record.call_id:
                lines.append(f"   call_id: {record.call_id}")
            if record.value:
                payload_dump = json.dumps(record.value, ensure_ascii=False)
                lines.append(f"   payload: {payload_dump}")
        return "\n".join(lines)

    def render_plan_snapshot(self) -> str:
        plan = select_latest(self._session, Plan)
        if plan is None:
            return "No active plan."

        lines = [f"Objective: {plan.objective} (status: {plan.status})"]
        for step in plan.steps:
            notes = "; ".join(step.notes) if step.notes else ""
            suffix = f" — notes: {notes}" if notes else ""
            if step.details:
                suffix = f" — details: {step.details}{suffix}"
            lines.append(f"- {step.step_id} [{step.status}] {step.title}{suffix}")
        return "\n".join(lines)

    def _on_tool_invoked(self, event: object) -> None:
        if not isinstance(event, ToolInvoked):
            return

        serialized_params = dump(event.params, exclude_none=True)
        payload = dump(event.result.value, exclude_none=True)
        print(
            f"[tool] {event.name} called with {serialized_params}\n"
            f"       → {event.result.message}"
        )
        if payload:
            print(f"       payload: {payload}")

        value = payload if isinstance(payload, dict) else {}
        record = ToolCallLog(
            name=event.name,
            prompt_name=event.prompt_name,
            message=event.result.message,
            value=value,
            call_id=event.call_id,
        )
        self._history.append(record)


def build_adapter() -> SupportsReviewEvaluate:
    provider = os.getenv("CODE_REVIEW_EXAMPLE_PROVIDER", "openai").strip().lower()
    if provider == "openai":
        if "OPENAI_API_KEY" not in os.environ:
            raise SystemExit("Set OPENAI_API_KEY before running this example.")
        model = os.getenv("OPENAI_MODEL", "gpt-5")
        return OpenAIAdapter(model=model)
    if provider == "litellm":
        api_key = os.getenv("LITELLM_API_KEY")
        if api_key is None:
            raise SystemExit("Set LITELLM_API_KEY before running this example.")
        completion_kwargs: dict[str, Any] = {"api_key": api_key}
        base_url = os.getenv("LITELLM_BASE_URL")
        if base_url:
            completion_kwargs["api_base"] = base_url
        model = os.getenv("LITELLM_MODEL", "gpt-5")
        return LiteLLMAdapter(model=model, completion_kwargs=completion_kwargs)
    raise SystemExit(
        "Supported providers: 'openai' (default) or 'litellm'."
        " Set CODE_REVIEW_EXAMPLE_PROVIDER accordingly."
    )


def main() -> None:
    print("Launching sunfish code review session with planning and VFS tooling...")
    print("- test-repositories/sunfish mounted under virtual path 'sunfish/'.")
    print(
        "- Available commands: 'history' (tool log), 'plan' (current plan),"
        " 'exit' to quit."
    )

    session = SunfishReviewSession(build_adapter())
    print("Type a review prompt to begin.")
    while True:
        try:
            prompt = input("Review prompt: ").strip()
        except EOFError:  # pragma: no cover - interactive convenience
            break
        if not prompt:
            print("Goodbye.")
            break
        lowered = prompt.lower()
        if lowered in {"exit", "quit"}:
            break
        if lowered == "history":
            print(session.render_tool_history())
            continue
        if lowered == "plan":
            print(session.render_plan_snapshot())
            continue
        answer = session.evaluate(prompt)
        print(f"Agent: {answer}")


if __name__ == "__main__":
    main()
