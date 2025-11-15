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

"""Interactive walkthrough showcasing a minimalist code review agent."""

from __future__ import annotations

import json
import logging
import os
import sys
import textwrap
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MethodType
from typing import Any, Protocol, cast

from weakincentives.adapters import PromptResponse
from weakincentives.adapters.core import SessionProtocol
from weakincentives.adapters.openai import OpenAIAdapter
from weakincentives.prompt import MarkdownSection, Prompt, SupportsDataclass
from weakincentives.prompt.overrides import (
    LocalPromptOverridesStore,
    PromptOverridesError,
)
from weakincentives.prompt.prompt import RenderedPrompt
from weakincentives.runtime.events import (
    EventBus,
    InProcessEventBus,
    PromptRendered,
    ToolInvoked,
)
from weakincentives.runtime.session import (
    ReducerContextProtocol,
    ReducerEvent,
    Session,
    append,
    select_latest,
)
from weakincentives.serde import dump
from weakincentives.tools import SubagentsSection
from weakincentives.tools.asteval import AstevalSection
from weakincentives.tools.planning import Plan, PlanningStrategy, PlanningToolsSection
from weakincentives.tools.vfs import HostMount, VfsPath, VfsToolsSection

PROJECT_ROOT = Path(__file__).resolve().parent
TEST_REPOSITORIES_ROOT = (PROJECT_ROOT / "test-repositories").resolve()
PROMPT_OVERRIDES_TAG_ENV = "CODE_REVIEW_PROMPT_TAG"
_DEFAULT_OVERRIDE_TAG = "latest"
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
)
SUNFISH_MOUNT_MAX_BYTES = 600_000
_LOG_STRING_LIMIT = 256


@dataclass(slots=True, frozen=True)
class ReviewGuidance:
    focus: str = field(
        default=(
            "Identify potential issues, risks, and follow-up questions for the changes "
            "under review."
        ),
        metadata={
            "description": "Default framing instructions for the review assistant.",
        },
    )


@dataclass(slots=True, frozen=True)
class ReviewTurnParams:
    request: str = field(
        metadata={
            "description": "User-provided review task or question to address.",
        }
    )


@dataclass(slots=True, frozen=True)
class ReviewResponse:
    summary: str
    issues: list[str]
    next_steps: list[str]


class SupportsReviewEvaluate(Protocol):
    def evaluate(
        self,
        prompt: Prompt[ReviewResponse],
        *params: SupportsDataclass,
        parse_output: bool = True,
        bus: EventBus,
        session: SessionProtocol,
    ) -> PromptResponse[ReviewResponse]: ...


class CodeReviewApp:
    """Owns adapter lifecycle, prompt overrides, and the REPL loop."""

    def __init__(
        self,
        adapter: SupportsReviewEvaluate,
        *,
        override_tag: str | None = None,
        overrides_store: LocalPromptOverridesStore | None = None,
    ) -> None:
        self.adapter = adapter
        self.override_tag = _resolve_override_tag(override_tag)
        self.overrides_store = overrides_store or LocalPromptOverridesStore()
        (
            self.prompt,
            self.session,
            self.bus,
        ) = build_code_reviewer_state(
            overrides_store=self.overrides_store,
            override_tag=self.override_tag,
        )

    def run(self) -> None:
        """Start the interactive review session."""

        print(_build_intro(self.override_tag))
        print("Type a review prompt to begin. (Type 'exit' to quit.)")
        while True:
            try:
                user_prompt = input("Review prompt: ").strip()
            except EOFError:  # pragma: no cover - interactive convenience
                print()
                break
            if not user_prompt:
                break
            if user_prompt.lower() in {"exit", "quit"}:
                break
            answer = self._evaluate_turn(user_prompt)
            print(f"Agent: {answer}")
            print("Plan snapshot:")
            print(_render_plan_snapshot(self.session))
        print("Goodbye.")

    def _evaluate_turn(self, user_prompt: str) -> str:
        response = self.adapter.evaluate(
            self.prompt,
            ReviewTurnParams(request=user_prompt),
            bus=self.bus,
            session=self.session,
        )
        return _render_response_payload(response)


def main() -> None:
    """Entry point used by the Codex CLI harness."""

    _configure_logging()
    adapter = build_adapter()
    app = CodeReviewApp(adapter)
    app.run()


def build_adapter() -> SupportsReviewEvaluate:
    if "OPENAI_API_KEY" not in os.environ:
        raise SystemExit("Set OPENAI_API_KEY before running this example.")
    model = os.getenv("OPENAI_MODEL", "gpt-5")
    return cast(SupportsReviewEvaluate, OpenAIAdapter(model=model))


def build_task_prompt(session: Session) -> Prompt[ReviewResponse]:
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
            - Python evaluation tools run short scripts with access to staged VFS
              reads and writes for quick experiments.

            Respond with JSON containing:
            - summary: One paragraph describing your findings so far.
            - issues: List concrete risks, questions, or follow-ups you found.
            - next_steps: Actionable recommendations to progress the task.
            """
        ).strip(),
        default_params=ReviewGuidance(),
        key="code-review-brief",
    )
    planning_section = PlanningToolsSection(
        strategy=PlanningStrategy.PLAN_ACT_REFLECT,
    )
    subagents_section = SubagentsSection()
    vfs_section = VfsToolsSection(
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
    asteval_section = AstevalSection()
    user_turn_section = MarkdownSection[ReviewTurnParams](
        title="Review Request",
        template="${request}",
        key="review-request",
    )
    return Prompt[ReviewResponse](
        ns="examples/code-review",
        key="code-review-session",
        name="sunfish_code_review_agent",
        sections=(
            guidance_section,
            planning_section,
            subagents_section,
            vfs_section,
            asteval_section,
            user_turn_section,
        ),
    )


def build_code_reviewer_state(
    *,
    overrides_store: LocalPromptOverridesStore | None = None,
    override_tag: str | None = None,
) -> tuple[Prompt[ReviewResponse], Session, EventBus]:
    """Initialize the prompt, session, and bus used by the interactive example."""

    store = overrides_store or LocalPromptOverridesStore()
    resolved_tag = _resolve_override_tag(override_tag)
    bus = InProcessEventBus()
    session = Session(bus=bus)
    session.register_reducer(PromptRendered, append)
    session.register_reducer(PromptRendered, _print_rendered_prompt)
    base_prompt = build_task_prompt(session)
    try:
        store.seed_if_necessary(base_prompt, tag=resolved_tag)
    except PromptOverridesError as exc:  # pragma: no cover - startup validation
        raise SystemExit(f"Failed to initialize prompt overrides: {exc}") from exc

    def render_with_session_overrides(
        prompt_obj: Prompt[Any],
        *params: SupportsDataclass,
        inject_output_instructions: bool | None = None,
    ) -> RenderedPrompt[Any]:
        return prompt_obj.render_with_overrides(
            *params,
            overrides_store=store,
            tag=resolved_tag,
            inject_output_instructions=inject_output_instructions,
        )

    prompt_with_any = cast(Any, base_prompt)
    prompt_with_any.render = MethodType(render_with_session_overrides, base_prompt)
    bus.subscribe(ToolInvoked, _log_tool_invocation)
    return base_prompt, session, bus


def _build_intro(override_tag: str) -> str:
    return textwrap.dedent(
        f"""
        Launching example code reviewer agent.
        - test-repositories/sunfish mounted under virtual path 'sunfish/'.
        - Tools: planning, subagents, VFS, and Python evaluation.
        - Command: 'exit' to quit.
        - Prompt overrides tag: '{override_tag}' (set {PROMPT_OVERRIDES_TAG_ENV} to change).
        """
    ).strip()


def _render_plan_snapshot(session: Session) -> str:
    plan = select_latest(session, Plan)
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


def _render_response_payload(response: PromptResponse[ReviewResponse]) -> str:
    if response.output is not None:
        rendered_output = dump(response.output, exclude_none=True)
        return json.dumps(rendered_output, ensure_ascii=False, indent=2)
    if response.text:
        return response.text
    return "(no response from assistant)"


def _resolve_override_tag(tag: str | None = None) -> str:
    candidate = tag or os.getenv(PROMPT_OVERRIDES_TAG_ENV, _DEFAULT_OVERRIDE_TAG)
    normalized = candidate.strip()
    return normalized or _DEFAULT_OVERRIDE_TAG


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def _print_rendered_prompt(
    slice_values: tuple[PromptRendered, ...],
    event: ReducerEvent,
    *,
    context: ReducerContextProtocol,
) -> tuple[PromptRendered, ...]:
    del context
    prompt_event = cast(PromptRendered, event)
    prompt_label = prompt_event.prompt_name or (
        f"{prompt_event.prompt_ns}:{prompt_event.prompt_key}"
    )
    print(f"[prompt] Rendered prompt ({prompt_label})")
    print(prompt_event.rendered_prompt)
    print()
    return slice_values


def _log_tool_invocation(event: object) -> None:
    tool_event = cast(ToolInvoked, event)
    params_repr = _format_for_log(dump(tool_event.params, exclude_none=True))
    result_message = _truncate_for_log(tool_event.result.message or "")
    payload_repr: str | None = None
    payload = tool_event.result.value
    if payload is not None:
        try:
            payload_repr = _format_for_log(dump(payload, exclude_none=True))
        except TypeError:
            payload_repr = _format_for_log({"value": payload})

    lines = [
        f"{tool_event.name} ({tool_event.prompt_name})",
        f"  params: {params_repr}",
        f"  result: {result_message}",
    ]
    if payload_repr is not None:
        lines.append(f"  payload: {payload_repr}")

    print("[tool] " + "\n".join(lines))


def _truncate_for_log(text: str, *, limit: int = _LOG_STRING_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return f"{text[: limit - 1]}…"


def _format_for_log(payload: object, *, limit: int = _LOG_STRING_LIMIT) -> str:
    serializable = _coerce_for_log(payload)
    try:
        rendered = json.dumps(serializable, ensure_ascii=False)
    except TypeError:
        rendered = repr(serializable)
    return _truncate_for_log(rendered, limit=limit)


def _coerce_for_log(payload: object) -> object:
    if payload is None or isinstance(payload, (str, int, float, bool)):
        return payload
    if isinstance(payload, Mapping):
        return {str(key): _coerce_for_log(value) for key, value in payload.items()}
    if isinstance(payload, Sequence) and not isinstance(
        payload,
        (str, bytes, bytearray),
    ):
        return [_coerce_for_log(item) for item in payload]
    if hasattr(payload, "__dataclass_fields__"):
        return dump(payload, exclude_none=True)
    if isinstance(payload, set):
        return sorted(_coerce_for_log(item) for item in payload)
    return str(payload)


if __name__ == "__main__":
    main()
