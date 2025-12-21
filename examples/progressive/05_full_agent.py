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

"""Full agent example: interactive REPL with MainLoop orchestration.

This is the culmination of the progressive examples, showing a complete
production-style agent. It demonstrates:
- MainLoop for standardized request/response orchestration
- Event bus for observability (logging prompt renders, tool calls, tokens)
- Progressive disclosure (sections start summarized, expand on demand)
- Persistent session across multiple REPL turns
- Deadline enforcement
- Structured output validation

This example uses the same patterns as the code_reviewer_example.py but
in a simpler domain to keep the code focused on the patterns.

Run with: uv run python examples/progressive/05_full_agent.py
Requires: OPENAI_API_KEY environment variable
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast, override

from examples import build_logged_session, configure_logging, render_plan_snapshot
from weakincentives import MarkdownSection, Prompt
from weakincentives.adapters.core import ProviderAdapter
from weakincentives.adapters.openai import OpenAIAdapter
from weakincentives.contrib.tools import (
    HostMount,
    PlanningStrategy,
    PlanningToolsSection,
    VfsPath,
    VfsToolsSection,
)
from weakincentives.deadlines import Deadline
from weakincentives.prompt import PromptTemplate, SectionVisibility
from weakincentives.runtime import (
    EventBus,
    InProcessEventBus,
    MainLoop,
    MainLoopCompleted,
    Session,
)

DEFAULT_DEADLINE_MINUTES = 5


# --- Structured Output ---


@dataclass(slots=True, frozen=True)
class AssistantResponse:
    """Response from the assistant for each turn."""

    summary: str = field(
        metadata={"description": "Summary of what was done or discovered."}
    )
    findings: list[str] = field(
        metadata={"description": "Key observations or results."}
    )
    next_steps: list[str] = field(
        metadata={"description": "Suggested follow-up actions."}
    )


# --- Request Parameters ---


@dataclass(slots=True, frozen=True)
class UserRequest:
    """User's request for each turn."""

    request: str = field(metadata={"description": "The user's request."})


# --- Reference Documentation (Progressive Disclosure) ---


@dataclass(slots=True, frozen=True)
class GuidanceParams:
    """Empty params for guidance section (required for typing)."""

    pass


@dataclass(slots=True, frozen=True)
class ReferenceParams:
    """Parameters for reference documentation section."""

    project_name: str = "sample_project"


# --- Sample Project ---


def create_sample_project(base_dir: Path) -> None:
    """Create a sample project for the assistant to explore."""
    src = base_dir / "src"
    src.mkdir(parents=True, exist_ok=True)

    _ = (src / "app.py").write_text('''\
"""Main application module."""

from .database import get_connection
from .handlers import handle_request


def main():
    """Entry point."""
    conn = get_connection()
    # TODO: Add proper error handling
    result = handle_request(conn, {"action": "greet"})
    print(result)


if __name__ == "__main__":
    main()
''')

    _ = (src / "database.py").write_text('''\
"""Database connection management."""

import sqlite3
from typing import Optional

_connection: Optional[sqlite3.Connection] = None


def get_connection() -> sqlite3.Connection:
    """Get or create database connection."""
    global _connection
    if _connection is None:
        _connection = sqlite3.connect(":memory:")
    return _connection


def close_connection() -> None:
    """Close the database connection."""
    global _connection
    if _connection is not None:
        _connection.close()
        _connection = None
''')

    _ = (src / "handlers.py").write_text('''\
"""Request handlers."""

from typing import Any


def handle_request(conn: Any, request: dict) -> str:
    """Handle incoming requests."""
    action = request.get("action", "unknown")

    if action == "greet":
        return "Hello, World!"
    elif action == "status":
        return "OK"
    else:
        return f"Unknown action: {action}"
''')

    _ = (base_dir / "README.md").write_text("""\
# Sample Project

A demonstration application for the full agent example.

## Modules

- `src/app.py` - Main entry point
- `src/database.py` - Database connection management
- `src/handlers.py` - Request handling logic

## Running

```bash
python -m src.app
```
""")

    _ = (base_dir / "requirements.txt").write_text("sqlite3\n")


# --- MainLoop Implementation ---


class AssistantLoop(MainLoop[UserRequest, AssistantResponse]):
    """MainLoop implementation for the interactive assistant.

    Maintains a persistent session across all turns and provides
    the standardized execute() workflow.
    """

    _session: Session
    _template: PromptTemplate[AssistantResponse]

    def __init__(
        self,
        *,
        adapter: ProviderAdapter[AssistantResponse],
        bus: EventBus,
        project_dir: Path,
    ) -> None:
        super().__init__(adapter=adapter, bus=bus)
        # Create persistent session at construction time
        self._session = build_logged_session(tags={"app": "assistant"})
        self._template = self._build_template(project_dir)

    def _build_template(self, project_dir: Path) -> PromptTemplate[AssistantResponse]:
        """Build the prompt template with all sections."""
        return PromptTemplate[AssistantResponse](
            ns="examples/progressive",
            key="full-agent",
            name="assistant_agent",
            sections=(
                self._build_guidance_section(),
                self._build_reference_section(),
                PlanningToolsSection(
                    session=self._session,
                    strategy=PlanningStrategy.PLAN_ACT_REFLECT,
                ),
                VfsToolsSection(
                    session=self._session,
                    mounts=(
                        HostMount(
                            host_path=str(project_dir),
                            mount_path=VfsPath(("project",)),
                            include_glob=("*.py", "*.md", "*.txt"),
                            max_bytes=100_000,
                        ),
                    ),
                    allowed_host_roots=(str(project_dir.parent),),
                ),
                MarkdownSection[UserRequest](
                    title="User Request",
                    template="${request}",
                    key="user-request",
                ),
            ),
        )

    @staticmethod
    def _build_guidance_section() -> MarkdownSection[GuidanceParams]:
        """Build the main guidance section."""
        return MarkdownSection[GuidanceParams](
            title="Assistant Instructions",
            template="""\
You are a helpful code assistant. The project is mounted under `project/`.

Use the available tools to:
- Create and update plans to track your work
- Explore files with ls, read_file, grep, and glob
- Answer questions about the codebase

Respond with JSON containing:
- summary: What you did or discovered
- findings: Key observations (list of strings)
- next_steps: Recommended follow-up actions (list of strings)
""",
            key="guidance",
            default_params=GuidanceParams(),
        )

    @staticmethod
    def _build_reference_section() -> MarkdownSection[ReferenceParams]:
        """Build reference docs with progressive disclosure.

        This section starts summarized. The LLM can request expansion
        by calling open_sections if it needs the full documentation.
        """
        return MarkdownSection[ReferenceParams](
            title="Reference Documentation",
            template="""\
## Project: ${project_name}

### Code Conventions
- Use type hints for all function signatures
- Add docstrings to public functions
- Handle errors explicitly, don't silently fail
- Keep functions focused and small

### Common Patterns
- Database connections should be managed via context managers
- Request handlers should validate input before processing
- Global state should be avoided where possible

### Review Checklist
- Are there any TODO comments that need addressing?
- Is error handling comprehensive?
- Are there potential security issues?
- Is the code well-documented?
""",
            summary="Reference documentation for ${project_name} is available. Request expansion if needed.",
            key="reference-docs",
            visibility=SectionVisibility.SUMMARY,
            default_params=ReferenceParams(),
        )

    @override
    def create_prompt(self, request: UserRequest) -> Prompt[AssistantResponse]:
        """Create and bind the prompt for the given request."""
        return Prompt(self._template).bind(request)

    @override
    def create_session(self) -> Session:
        """Return the persistent session (reused across all turns)."""
        return self._session

    @property
    def session(self) -> Session:
        """Expose session for external inspection."""
        return self._session


# --- Application ---


class AssistantApp:
    """Owns the REPL loop and user interaction."""

    _bus: InProcessEventBus
    _loop: AssistantLoop

    def __init__(
        self,
        adapter: ProviderAdapter[AssistantResponse],
        project_dir: Path,
    ) -> None:
        super().__init__()
        self._bus = InProcessEventBus()
        self._loop = AssistantLoop(
            adapter=adapter,
            bus=self._bus,
            project_dir=project_dir,
        )
        self._bus.subscribe(MainLoopCompleted, self._on_loop_completed)

    def _on_loop_completed(self, event: object) -> None:
        """Handle completed response from event bus."""
        completed: MainLoopCompleted[AssistantResponse] = cast(
            MainLoopCompleted[AssistantResponse], event
        )
        response = completed.response

        print("\n" + "=" * 50)
        print("ASSISTANT RESPONSE")
        print("=" * 50)

        if response.output is not None:
            output = response.output
            print(f"\nSummary: {output.summary}")

            if output.findings:
                print("\nFindings:")
                for finding in output.findings:
                    print(f"  - {finding}")

            if output.next_steps:
                print("\nNext steps:")
                for step in output.next_steps:
                    print(f"  - {step}")
        else:
            print(f"\n{response.text or '(no response)'}")

        print("\n" + "-" * 50)
        print("Plan Snapshot:")
        print("-" * 50)
        print(render_plan_snapshot(self._loop.session))
        print()

    def run(self) -> None:
        """Start the interactive assistant session."""
        print("=" * 60)
        print("Interactive Assistant (Full Agent Example)")
        print("=" * 60)
        print("\nThis assistant can explore the mounted project and answer")
        print("questions about the code. Type 'exit' or 'quit' to stop.\n")

        try:
            while True:
                try:
                    user_input = input("You: ").strip()
                except EOFError:
                    print()
                    break

                if not user_input or user_input.lower() in {"exit", "quit"}:
                    break

                # Execute with deadline
                deadline = Deadline(
                    expires_at=datetime.now(UTC)
                    + timedelta(minutes=DEFAULT_DEADLINE_MINUTES)
                )

                request = UserRequest(request=user_input)
                try:
                    _ = self._loop.execute(request, deadline=deadline)
                except Exception as e:
                    print(f"\nError: {e}\n")

        except KeyboardInterrupt:
            print("\n\nInterrupted.")

        print("\nGoodbye!")


def main() -> None:
    """Entry point for the full agent example."""
    if "OPENAI_API_KEY" not in os.environ:
        raise SystemExit("Set OPENAI_API_KEY to run this example.")

    configure_logging()

    # Create a temporary project
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir) / "sample_project"
        project_dir.mkdir()
        create_sample_project(project_dir)

        print(f"\nCreated sample project at: {project_dir}")
        print("Files:")
        for path in sorted(project_dir.rglob("*")):
            if path.is_file():
                print(f"  {path.relative_to(project_dir)}")
        print()

        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        adapter = cast(
            ProviderAdapter[AssistantResponse],
            OpenAIAdapter(model=model),
        )

        app = AssistantApp(adapter, project_dir)
        app.run()


if __name__ == "__main__":
    main()
