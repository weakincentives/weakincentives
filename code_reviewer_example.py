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
from typing import cast
from uuid import UUID

from weakincentives.adapters import PromptResponse
from weakincentives.adapters.core import OptimizationScope, ProviderAdapter
from weakincentives.adapters.openai import OpenAIAdapter
from weakincentives.prompt import MarkdownSection, Prompt, SupportsDataclass
from weakincentives.prompt.overrides import (
    LocalPromptOverridesStore,
    PromptOverridesError,
)
from weakincentives.runtime.events import EventBus, PromptRendered, ToolInvoked
from weakincentives.runtime.session import Session, select_latest
from weakincentives.serde import dump
from weakincentives.tools import SubagentsSection
from weakincentives.tools.digests import WorkspaceDigestSection
from weakincentives.tools.planning import Plan, PlanningStrategy, PlanningToolsSection
from weakincentives.tools.podman import PodmanSandboxSection
from weakincentives.tools.vfs import HostMount, VfsPath, VfsToolsSection

PROJECT_ROOT = Path(__file__).resolve().parent
TEST_REPOSITORIES_ROOT = (PROJECT_ROOT / "test-repositories").resolve()
PROMPT_OVERRIDES_TAG_ENV = "CODE_REVIEW_PROMPT_TAG"
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
_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class ReviewGuidance:
    """Default guidance for the review agent."""

    focus: str = field(
        default=(
            "Identify potential issues, risks, and follow-up questions for the changes "
            "under review."
        ),
        metadata={
            "description": "Default framing instructions for the review assistant."
        },
    )


@dataclass(slots=True, frozen=True)
class ReviewTurnParams:
    """Dataclass for dynamic parameters provided at runtime."""

    request: str = field(metadata={"description": "User-provided review request."})


@dataclass(slots=True, frozen=True)
class ReviewResponse:
    """Structured response emitted by the agent."""

    summary: str
    issues: list[str]
    next_steps: list[str]


@dataclass(slots=True)
class RuntimeContext:
    """Holds the prompt, session, and override handles for the REPL."""

    prompt: Prompt[ReviewResponse]
    session: Session
    bus: EventBus
    overrides_store: LocalPromptOverridesStore
    override_tag: str


class CodeReviewApp:
    """Owns adapter lifecycle, prompt overrides, and the REPL loop."""

    def __init__(
        self,
        adapter: ProviderAdapter[ReviewResponse],
        *,
        overrides_store: LocalPromptOverridesStore | None = None,
        override_tag: str | None = None,
    ) -> None:
        self.adapter = adapter
        self.context = _create_runtime_context(
            overrides_store=overrides_store,
            override_tag=override_tag,
        )
        self.prompt = self.context.prompt
        self.session = self.context.session
        self.bus = self.context.bus
        self.overrides_store = self.context.overrides_store
        self.override_tag = self.context.override_tag

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
            command, _space, _remainder = user_prompt.partition(" ")
            if command.lower() == "optimize":
                self._handle_optimize_command()
                continue
            if user_prompt.lower() in {"exit", "quit"}:
                break
            answer = self._evaluate_turn(user_prompt)
            print("\n--- Agent Response ---")
            print(answer)
            print("\n--- Plan Snapshot ---")
            print(_render_plan_snapshot(self.session))
            print("-" * 23 + "\n")

        print("Goodbye.")

    def _evaluate_turn(self, user_prompt: str) -> str:
        response = self.adapter.evaluate(
            self.prompt,
            ReviewTurnParams(request=user_prompt),
            bus=self.bus,
            session=self.session,
            overrides_store=self.overrides_store,
            overrides_tag=self.override_tag,
        )
        return _render_response_payload(response)

    def _handle_optimize_command(self) -> None:
        """Runs the optimization prompt and persists workspace digest content."""

        optimization_session = self._create_optimization_session()
        result = self.adapter.optimize(
            self.prompt,
            store_scope=OptimizationScope.SESSION,
            session=self.session,
            optimization_session=optimization_session,
        )
        digest = result.digest.strip()
        print("\nWorkspace digest persisted for future review turns:\n")
        print(digest)

    def _create_optimization_session(self) -> Session:
        """Provision a fresh session for optimization runs."""

        return _build_logged_session()


def main() -> None:
    """Entry point used by the `weakincentives` CLI harness."""

    _configure_logging()
    adapter = build_adapter()
    app = CodeReviewApp(adapter)
    app.run()


def build_adapter() -> ProviderAdapter[ReviewResponse]:
    """Build the OpenAI adapter, checking for the required API key."""

    if "OPENAI_API_KEY" not in os.environ:
        raise SystemExit("Set OPENAI_API_KEY before running this example.")
    model = os.getenv("OPENAI_MODEL", "gpt-5.1")
    return cast(ProviderAdapter[ReviewResponse], OpenAIAdapter(model=model))


def build_task_prompt(*, session: Session) -> Prompt[ReviewResponse]:
    """Builds the main prompt for the code review agent."""

    _ensure_test_repository_available()
    workspace_section = _build_workspace_section(session=session)
    sections = (
        _build_review_guidance_section(),
        WorkspaceDigestSection(session=session),
        _build_subagents_section(),
        PlanningToolsSection(
            session=session, strategy=PlanningStrategy.PLAN_ACT_REFLECT
        ),
        workspace_section,
        MarkdownSection[ReviewTurnParams](
            title="Review Request",
            template="${request}",
            key="review-request",
        ),
    )
    return Prompt[ReviewResponse](
        ns="examples/code-review",
        key="code-review-session",
        name="sunfish_code_review_agent",
        sections=sections,
    )


def _ensure_test_repository_available() -> None:
    if TEST_REPOSITORIES_ROOT.exists():
        return
    raise SystemExit(
        f"Expected test repositories under {TEST_REPOSITORIES_ROOT!s},"
        " but the directory is missing."
    )


def _build_review_guidance_section() -> MarkdownSection[ReviewGuidance]:
    return MarkdownSection[ReviewGuidance](
        title="Code Review Brief",
        template=textwrap.dedent(
            """
            You are a code review assistant exploring the mounted workspace.
            Access the repository under the `sunfish/` directory.

            Use the available tools to stay grounded:
            - Planning tools help you capture multi-step investigations; keep the
              plan updated as you explore.
            - Filesystem tools list directories, read files, and stage edits.
              When available, the `shell_execute` command runs short Podman
              commands (no network access). Mounted files are read-only; use
              writes to stage new snapshots.
            - `dispatch_subagents` lets you delegate parallel scans (e.g., README,
              docs, build scripts) so you can summarize broader surface area faster.

            Respond with JSON containing:
            - summary: One paragraph describing your findings so far.
            - issues: List concrete risks, questions, or follow-ups you found.
            - next_steps: Actionable recommendations to progress the task.
            """
        ).strip(),
        default_params=ReviewGuidance(),
        key="code-review-brief",
    )


def _build_subagents_section() -> SubagentsSection:
    return SubagentsSection()


def _sunfish_mounts() -> tuple[HostMount, ...]:
    return (
        HostMount(
            host_path="sunfish",
            mount_path=VfsPath(("sunfish",)),
            include_glob=SUNFISH_MOUNT_INCLUDE_GLOBS,
            exclude_glob=SUNFISH_MOUNT_EXCLUDE_GLOBS,
            max_bytes=SUNFISH_MOUNT_MAX_BYTES,
        ),
    )


def _build_workspace_section(
    *,
    session: Session,
) -> MarkdownSection[SupportsDataclass]:
    mounts = _sunfish_mounts()
    allowed_roots = (TEST_REPOSITORIES_ROOT,)
    connection = PodmanSandboxSection.resolve_connection()
    if connection is None:
        _LOGGER.info(
            "Podman connection unavailable; falling back to VFS tools for the code reviewer example."
        )
        return VfsToolsSection(
            session=session,
            mounts=mounts,
            allowed_host_roots=allowed_roots,
        )

    return PodmanSandboxSection(
        session=session,
        mounts=mounts,
        allowed_host_roots=allowed_roots,
        base_url=connection.get("base_url"),
        identity=connection.get("identity"),
        connection_name=connection.get("connection_name"),
    )


def initialize_code_reviewer_runtime(
    *,
    overrides_store: LocalPromptOverridesStore | None = None,
    override_tag: str | None = None,
) -> tuple[
    Prompt[ReviewResponse],
    Session,
    EventBus,
    LocalPromptOverridesStore,
    str,
]:
    context = _create_runtime_context(
        overrides_store=overrides_store,
        override_tag=override_tag,
    )
    return (
        context.prompt,
        context.session,
        context.bus,
        context.overrides_store,
        context.override_tag,
    )


def _create_runtime_context(
    *,
    overrides_store: LocalPromptOverridesStore | None = None,
    override_tag: str | None = None,
) -> RuntimeContext:
    store = overrides_store or LocalPromptOverridesStore()
    session = _build_logged_session()
    bus = session.event_bus
    resolved_tag = _resolve_override_tag(override_tag, session_id=session.session_id)
    prompt = build_task_prompt(session=session)
    try:
        store.seed_if_necessary(prompt, tag=resolved_tag)
    except PromptOverridesError as exc:  # pragma: no cover - startup validation
        raise SystemExit(f"Failed to initialize prompt overrides: {exc}") from exc

    return RuntimeContext(
        prompt=prompt,
        session=session,
        bus=bus,
        overrides_store=store,
        override_tag=resolved_tag,
    )


def _build_logged_session() -> Session:
    session = Session()
    _attach_logging_subscribers(session.event_bus)
    return session


def _build_intro(override_tag: str) -> str:
    return textwrap.dedent(
        f"""
        Launching example code reviewer agent.
        - Repository: test-repositories/sunfish mounted under virtual path 'sunfish/'.
        - Tools: Planning and a filesystem workspace (with a Podman shell if available).
        - Overrides: Using tag '{override_tag}' (set {PROMPT_OVERRIDES_TAG_ENV} to change).
        - Commands: 'optimize' refreshes the workspace digest, 'exit' quits.

        Note: Full prompt text and tool calls will be logged to the console for observability.
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
        output = response.output
        lines = [f"Summary: {output.summary}"]
        if output.issues:
            lines.append("Issues:")
            lines.extend(f"- {issue}" for issue in output.issues)
        if output.next_steps:
            lines.append("Next Steps:")
            lines.extend(f"- {step}" for step in output.next_steps)
        return "\n".join(lines)
    if response.text:
        return response.text
    return "(no response from assistant)"


def _resolve_override_tag(
    tag: str | None,
    *,
    session_id: UUID,
) -> str:
    if tag is not None:
        normalized = tag.strip()
        if normalized:
            return normalized
    env_candidate = os.getenv(PROMPT_OVERRIDES_TAG_ENV, "").strip()
    if env_candidate:
        return env_candidate
    return "latest"


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def _print_rendered_prompt(event: object) -> None:
    prompt_event = cast(PromptRendered, event)
    prompt_label = prompt_event.prompt_name or (
        f"{prompt_event.prompt_ns}:{prompt_event.prompt_key}"
    )
    print(f"\n[prompt] Rendered prompt ({prompt_label})")
    print(prompt_event.rendered_prompt)
    print()


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

    print("\n[tool] " + "\n".join(lines))


def _attach_logging_subscribers(bus: EventBus) -> None:
    bus.subscribe(PromptRendered, _print_rendered_prompt)
    bus.subscribe(ToolInvoked, _log_tool_invocation)


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
        payload, (str, bytes, bytearray)
    ):
        return [_coerce_for_log(item) for item in payload]
    if hasattr(payload, "__dataclass_fields__"):
        return dump(payload, exclude_none=True)
    if isinstance(payload, set):
        return sorted(_coerce_for_log(item) for item in payload)
    return str(payload)


if __name__ == "__main__":
    main()
