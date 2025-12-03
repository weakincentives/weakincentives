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

"""Guided walkthrough for a minimalist code review agent."""

from __future__ import annotations

import json
import logging
import os
import sys
import textwrap
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from uuid import UUID

from weakincentives.adapters import PromptResponse, ProviderAdapter
from weakincentives.adapters.core import OptimizationScope
from weakincentives.adapters.openai import OpenAIAdapter
from weakincentives.debug import dump_session as dump_session_tree
from weakincentives.prompt import MarkdownSection, Prompt, PromptTemplate
from weakincentives.prompt.overrides import (
    LocalPromptOverridesStore,
    PromptOverridesError,
)
from weakincentives.runtime import (
    EventBus,
    PromptExecuted,
    PromptRendered,
    Session,
    ToolInvoked,
    select_latest,
)
from weakincentives.serde import dump
from weakincentives.tools import (
    HostMount,
    Plan,
    PlanningStrategy,
    PlanningToolsSection,
    PodmanSandboxConfig,
    PodmanSandboxSection,
    SubagentsSection,
    VfsPath,
    VfsToolsSection,
    WorkspaceDigestSection,
)

PROJECT_ROOT = Path(__file__).resolve().parent
TEST_REPOSITORIES_ROOT = (PROJECT_ROOT / "test-repositories").resolve()
SNAPSHOT_DIR = PROJECT_ROOT / "snapshots"
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
class ReviewTurnParams:
    """Request text supplied by the person driving the walkthrough."""

    request: str


@dataclass(slots=True, frozen=True)
class ReviewResponse:
    """Structured reply produced by the code review prompt."""

    summary: str
    issues: list[str]
    next_steps: list[str]


def main() -> None:
    """Entrypoint invoked by the repository's CLI harness."""

    _configure_logging()
    adapter = build_adapter()
    run_review_walkthrough(adapter)


# -- Tutorial steps ---------------------------------------------------------


def run_review_walkthrough(
    adapter: ProviderAdapter[ReviewResponse],
    *,
    overrides_store: LocalPromptOverridesStore | None = None,
    override_tag: str | None = None,
) -> None:
    """Launch a REPL that narrates each move the example makes.

    The walkthrough builds a prompt, wires in overrides, attaches event
    subscribers so that prompt renders and tool calls are echoed to stdout,
    and keeps running until the user types ``exit``.
    """

    prompt, session, bus, store, tag = prepare_runtime(
        overrides_store=overrides_store, override_tag=override_tag
    )

    print(_build_intro(tag))
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
        if user_prompt.split(" ", 1)[0].lower() == "optimize":
            digest = refresh_workspace_digest(
                prompt,
                adapter,
                session=session,
                overrides_store=store,
                override_tag=tag,
            )
            print("\nWorkspace digest persisted for future review turns:\n")
            print(digest)
            continue

        bound_prompt = prompt.bind(ReviewTurnParams(request=user_prompt))
        response = adapter.evaluate(bound_prompt, bus=bus, session=session)
        print("\n--- Agent Response ---")
        print(_render_response_payload(response))
        print("\n--- Plan Snapshot ---")
        print(_render_plan_snapshot(session))
        print("-" * 23 + "\n")

    print("Goodbye.")
    dump_session_tree(session, SNAPSHOT_DIR)


def prepare_runtime(
    *,
    overrides_store: LocalPromptOverridesStore | None = None,
    override_tag: str | None = None,
) -> tuple[Prompt[ReviewResponse], Session, EventBus, LocalPromptOverridesStore, str]:
    """Assemble the prompt, overrides, and logging hooks used by the REPL."""

    _ensure_test_repository_available()
    store = overrides_store or LocalPromptOverridesStore()
    session = _build_logged_session(tags={"app": "code-reviewer"})
    bus = session.event_bus
    resolved_tag = _resolve_override_tag(override_tag, session_id=session.session_id)
    prompt_template = build_review_prompt(session=session)
    prompt = Prompt(prompt_template, overrides_store=store, overrides_tag=resolved_tag)

    try:
        store.seed(prompt_template, tag=resolved_tag)
    except PromptOverridesError as exc:  # pragma: no cover - startup validation
        raise SystemExit(f"Failed to initialize prompt overrides: {exc}") from exc

    return prompt, session, bus, store, resolved_tag


def refresh_workspace_digest(
    prompt: Prompt[ReviewResponse],
    adapter: ProviderAdapter[ReviewResponse],
    *,
    session: Session,
    overrides_store: LocalPromptOverridesStore,
    override_tag: str,
) -> str:
    """Ask the provider to regenerate the workspace digest and persist it."""

    optimization_session = _build_logged_session(parent=session)
    result = adapter.optimize(
        prompt,
        store_scope=OptimizationScope.SESSION,
        overrides_store=overrides_store,
        overrides_tag=override_tag,
        session=session,
        optimization_session=optimization_session,
    )
    return result.digest.strip()


# -- Prompt construction ----------------------------------------------------


def build_adapter() -> ProviderAdapter[ReviewResponse]:
    """Build the OpenAI adapter, checking for the required API key."""

    if "OPENAI_API_KEY" not in os.environ:
        raise SystemExit("Set OPENAI_API_KEY before running this example.")
    model = os.getenv("OPENAI_MODEL", "gpt-5.1")
    return cast(ProviderAdapter[ReviewResponse], OpenAIAdapter(model=model))


def build_review_prompt(*, session: Session) -> PromptTemplate[ReviewResponse]:
    """Compose the full prompt used by the walkthrough."""

    workspace_section = _build_workspace_section(session=session)
    return PromptTemplate[ReviewResponse](
        ns="examples/code-review",
        key="code-review-session",
        name="sunfish_code_review_agent",
        sections=(
            MarkdownSection(
                title="Code Review Brief",
                template=textwrap.dedent(
                    """
                    You are a code review assistant working in the `sunfish/` workspace.

                    Stay grounded with the tools:
                    - Planning captures objectives and updates.
                    - Filesystem tools list/read files; Podman shell runs short commands
                      when available (no network, mounts are read-only).
                    - `dispatch_subagents` delegates quick scans.

                    Respond with JSON fields:
                    - summary: Brief paragraph of findings.
                    - issues: Concrete risks, questions, or follow-ups.
                    - next_steps: Actionable recommendations.
                    """
                ).strip(),
                key="code-review-brief",
            ),
            WorkspaceDigestSection(session=session),
            SubagentsSection(),
            PlanningToolsSection(
                session=session, strategy=PlanningStrategy.PLAN_ACT_REFLECT
            ),
            workspace_section,
            MarkdownSection[ReviewTurnParams](
                title="Review Request",
                template="${request}",
                key="review-request",
            ),
        ),
    )


def _build_workspace_section(*, session: Session) -> MarkdownSection:
    mounts = (
        HostMount(
            host_path="sunfish",
            mount_path=VfsPath(("sunfish",)),
            include_glob=SUNFISH_MOUNT_INCLUDE_GLOBS,
            exclude_glob=SUNFISH_MOUNT_EXCLUDE_GLOBS,
            max_bytes=SUNFISH_MOUNT_MAX_BYTES,
        ),
    )
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
        config=PodmanSandboxConfig(
            mounts=mounts,
            allowed_host_roots=allowed_roots,
            base_url=connection.get("base_url"),
            identity=connection.get("identity"),
            connection_name=connection.get("connection_name"),
        ),
    )


# -- Logging helpers --------------------------------------------------------


def _build_logged_session(
    *, parent: Session | None = None, tags: Mapping[str, str] | None = None
) -> Session:
    session_tags: dict[str, str] = {"app": "code-reviewer"}
    if tags:
        session_tags.update(tags)

    session = Session(parent=parent, tags=cast(Mapping[object, object], session_tags))
    _attach_logging_subscribers(session.event_bus)
    return session


def _build_intro(override_tag: str) -> str:
    return textwrap.dedent(
        f"""
        Code reviewer ready.
        - Repo: test-repositories/sunfish mounted at 'sunfish/'.
        - Tools: Planning plus workspace access (Podman shell when available).
        - Overrides tag: '{override_tag}' (set {PROMPT_OVERRIDES_TAG_ENV} to change).
        - Commands: 'optimize' refreshes the digest, 'exit' quits.

        Prompts and tool calls are logged for visibility.
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


def _resolve_override_tag(tag: str | None, *, session_id: UUID) -> str:
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


def _log_prompt_executed(event: object) -> None:
    prompt_event = cast(PromptExecuted, event)
    usage = prompt_event.usage

    print("\n[prompt] Execution complete")
    if usage is None:
        print("  token usage: (not reported)\n")
        return

    parts: list[str] = []
    if usage.input_tokens is not None:
        parts.append(f"input={usage.input_tokens}")
    if usage.output_tokens is not None:
        parts.append(f"output={usage.output_tokens}")
    if usage.cached_tokens is not None:
        parts.append(f"cached={usage.cached_tokens}")

    total = usage.total_tokens
    if total is not None:
        parts.append(f"total={total}")

    print(f"  token usage: {', '.join(parts)}\n")


def _attach_logging_subscribers(bus: EventBus) -> None:
    bus.subscribe(PromptRendered, _print_rendered_prompt)
    bus.subscribe(ToolInvoked, _log_tool_invocation)
    bus.subscribe(PromptExecuted, _log_prompt_executed)


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
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        return [_coerce_for_log(item) for item in payload]
    if hasattr(payload, "__dataclass_fields__"):
        return dump(payload, exclude_none=True)
    if isinstance(payload, set):
        return sorted(_coerce_for_log(item) for item in payload)
    return str(payload)


def _ensure_test_repository_available() -> None:
    if TEST_REPOSITORIES_ROOT.exists():
        return
    raise SystemExit(
        f"Expected test repositories under {TEST_REPOSITORIES_ROOT!s},"
        " but the directory is missing.",
    )


if __name__ == "__main__":
    main()
