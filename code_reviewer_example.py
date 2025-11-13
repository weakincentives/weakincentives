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

"""Minimal interactive code reviewer showcasing the core toolkit."""

from __future__ import annotations

import json
import logging
import os
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from types import MethodType
from typing import Any, Protocol, cast

from weakincentives.adapters import PromptResponse
from weakincentives.adapters.core import SessionProtocol
from weakincentives.adapters.litellm import LiteLLMAdapter
from weakincentives.adapters.openai import OpenAIAdapter
from weakincentives.prompt import (
    MarkdownSection,
    Prompt,
    SupportsDataclass,
)
from weakincentives.prompt.overrides import (
    LocalPromptOverridesStore,
    PromptOverridesError,
)
from weakincentives.prompt.prompt import RenderedPrompt
from weakincentives.runtime.events import EventBus, InProcessEventBus, ToolInvoked
from weakincentives.runtime.session import Session, select_latest
from weakincentives.serde import dump
from weakincentives.tools import SubagentsSection
from weakincentives.tools.asteval import AstevalSection
from weakincentives.tools.planning import (
    Plan,
    PlanningStrategy,
    PlanningToolsSection,
)
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


def _truncate_for_log(text: str, *, limit: int = _LOG_STRING_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return f"{text[: limit - 1]}…"


def _format_for_log(payload: object, *, limit: int = _LOG_STRING_LIMIT) -> str:
    try:
        rendered = json.dumps(payload, ensure_ascii=False)
    except TypeError:
        rendered = repr(payload)
    return _truncate_for_log(rendered, limit=limit)


def _resolve_override_tag(tag: str | None = None) -> str:
    """Normalize the override tag used for prompt renders."""

    candidate = tag or os.getenv(PROMPT_OVERRIDES_TAG_ENV, _DEFAULT_OVERRIDE_TAG)
    normalized = candidate.strip()
    return normalized or _DEFAULT_OVERRIDE_TAG


def _configure_logging() -> None:
    """Ensure INFO level logging is emitted to stdout."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def main() -> None:
    """Launch the interactive code review walkthrough."""

    _configure_logging()
    override_tag = _resolve_override_tag()
    intro = textwrap.dedent(
        f"""
        Launching sunfish code review session.
        - test-repositories/sunfish mounted under virtual path 'sunfish/'.
        - Tools: planning, subagents, VFS, and Python evaluation.
        - Commands: 'history' (tool log), 'plan' (latest plan), 'exit' to quit.
        - Prompt overrides tag: '{override_tag}' (set {PROMPT_OVERRIDES_TAG_ENV} to change).
        """
    ).strip()
    print(intro)

    session = SunfishReviewSession(
        build_adapter(),
        override_tag=override_tag,
    )
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


def build_adapter() -> SupportsReviewEvaluate:
    provider = os.getenv("CODE_REVIEW_EXAMPLE_PROVIDER", "openai").strip().lower()
    if provider == "openai":
        if "OPENAI_API_KEY" not in os.environ:
            raise SystemExit("Set OPENAI_API_KEY before running this example.")
        model = os.getenv("OPENAI_MODEL", "gpt-5")
        return cast(SupportsReviewEvaluate, OpenAIAdapter(model=model))
    if provider == "litellm":
        api_key = os.getenv("LITELLM_API_KEY")
        if api_key is None:
            raise SystemExit("Set LITELLM_API_KEY before running this example.")
        completion_kwargs: dict[str, Any] = {"api_key": api_key}
        base_url = os.getenv("LITELLM_BASE_URL")
        if base_url:
            completion_kwargs["api_base"] = base_url
        model = os.getenv("LITELLM_MODEL", "gpt-5")
        return cast(
            SupportsReviewEvaluate,
            LiteLLMAdapter(model=model, completion_kwargs=completion_kwargs),
        )
    raise SystemExit(
        "Supported providers: 'openai' (default) or 'litellm'."
        " Set CODE_REVIEW_EXAMPLE_PROVIDER accordingly."
    )


class SunfishReviewSession:
    """Interactive session that records tool calls and plan snapshots."""

    def __init__(
        self,
        adapter: SupportsReviewEvaluate,
        *,
        overrides_store: LocalPromptOverridesStore | None = None,
        override_tag: str | None = None,
    ) -> None:
        self._adapter = adapter
        self._bus = InProcessEventBus()
        self._session = Session(bus=self._bus)
        base_prompt = build_sunfish_prompt(self._session)
        for section in base_prompt.sections:
            if isinstance(section, VfsToolsSection):
                snapshot = section.latest_snapshot()
                logging.info("VFS root directory: %s", snapshot.root_path)
        self._overrides_store = overrides_store or LocalPromptOverridesStore()
        self._override_tag = _resolve_override_tag(override_tag)
        try:
            self._overrides_store.seed_if_necessary(
                base_prompt,
                tag=self._override_tag,
            )
        except PromptOverridesError as exc:
            raise SystemExit(f"Failed to initialize prompt overrides: {exc}") from exc

        # Prompt overrides let you iterate on copy without touching versioned prompts;
        # see specs/PROMPTS.md for guardrails around tagging and persistence.
        def render_with_session_overrides(
            prompt_obj: Prompt[Any],
            *params: SupportsDataclass,
            inject_output_instructions: bool | None = None,
        ) -> RenderedPrompt[Any]:
            return prompt_obj.render_with_overrides(
                *params,
                overrides_store=self._overrides_store,
                tag=self._override_tag,
                inject_output_instructions=inject_output_instructions,
            )

        prompt_with_any = cast(Any, base_prompt)
        prompt_with_any.render = MethodType(
            render_with_session_overrides,
            base_prompt,
        )
        self._prompt = base_prompt
        self._tool_log: list[dict[str, str | None]] = []
        self._bus.subscribe(ToolInvoked, self._on_tool_invoked)

    def evaluate(self, request: str) -> str:
        self._session.reset()
        response = self._adapter.evaluate(
            self._prompt,
            ReviewTurnParams(request=request),
            bus=self._bus,
            session=self._session,
        )
        if response.output is not None:
            rendered_output = dump(response.output, exclude_none=True)
            return json.dumps(rendered_output, ensure_ascii=False, indent=2)
        if response.text:
            return response.text
        return "(no response from assistant)"

    def reset(self) -> None:
        """Clear session state before running a new turn."""

        self._session.reset()

    def render_tool_history(self) -> str:
        if not self._tool_log:
            return "No tool calls recorded yet."

        lines: list[str] = []
        for index, record in enumerate(self._tool_log, start=1):
            name = record["name"] or "?"
            prompt_name = record["prompt"] or "?"
            result = record["result"] or ""
            lines.append(f"{index}. {name} ({prompt_name}) → {result}")
            params = record.get("params")
            if params:
                lines.append(f"   params: {params}")
            payload = record.get("payload")
            if payload:
                lines.append(f"   payload: {payload}")
            call_id = record.get("call_id")
            if call_id:
                lines.append(f"   call_id: {call_id}")
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
        tool_event = cast(ToolInvoked, event)

        params_repr = _format_for_log(
            dump(tool_event.params, exclude_none=True),
        )
        message = _truncate_for_log(tool_event.result.message or "")
        payload = tool_event.result.value
        payload_repr: str | None
        if payload is None:
            payload_repr = None
        else:
            try:
                payload_repr = _format_for_log(
                    dump(payload, exclude_none=True),
                )
            except TypeError:
                payload_repr = _format_for_log({"value": payload})

        lines = [
            f"{tool_event.name} ({tool_event.prompt_name})",
            f"  params: {params_repr}",
            f"  result: {message}",
        ]
        if payload_repr is not None:
            lines.append(f"  payload: {payload_repr}")

        formatted = "\n".join(lines)
        print(f"[tool] {formatted}")
        self._tool_log.append(
            {
                "name": tool_event.name,
                "prompt": tool_event.prompt_name,
                "params": params_repr,
                "result": message,
                "payload": payload_repr,
                "call_id": tool_event.call_id,
            }
        )


def build_sunfish_prompt(session: Session) -> Prompt[ReviewResponse]:
    if not TEST_REPOSITORIES_ROOT.exists():
        raise SystemExit(
            f"Expected test repositories under {TEST_REPOSITORIES_ROOT!s},"
            " but the directory is missing."
        )

    # Prompt layout mirrors the design captured in specs/PROMPTS.md; update the spec
    # first when changing section ordering or required fields.
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


if __name__ == "__main__":
    main()
