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

"""Example with workspace: VFS file exploration.

Building on 03_with_session.py, this example adds a virtual filesystem
that the LLM can explore. It demonstrates:
- VfsToolsSection for file operations (ls, read, write, glob, grep)
- HostMount to mirror local directories into the VFS
- WorkspaceDigestSection for auto-generated workspace summaries
- Combining planning tools with file exploration

The VFS provides a sandboxed environment where the LLM can safely explore
files without accessing arbitrary paths on the host system.

Run with: uv run python examples/progressive/04_with_workspace.py
Requires: OPENAI_API_KEY environment variable
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from weakincentives import MarkdownSection, Prompt
from weakincentives.adapters.openai import OpenAIAdapter
from weakincentives.contrib.tools import (
    HostMount,
    Plan,
    PlanningStrategy,
    PlanningToolsSection,
    VfsPath,
    VfsToolsSection,
    WorkspaceDigestSection,
)
from weakincentives.prompt import PromptTemplate
from weakincentives.runtime import Session

# --- Structured Output ---


@dataclass(slots=True, frozen=True)
class ExplorationResult:
    """Result from exploring the workspace."""

    summary: str = field(
        metadata={"description": "What was discovered during exploration."}
    )
    files_examined: list[str] = field(
        metadata={"description": "List of files that were read."}
    )
    key_findings: list[str] = field(
        metadata={"description": "Important discoveries about the codebase."}
    )
    suggestions: list[str] = field(
        metadata={"description": "Recommendations for improvements."}
    )


# --- Prompt Parameters ---


@dataclass(slots=True, frozen=True)
class ExplorationParams:
    """Parameters for the exploration task."""

    task: str = field(metadata={"description": "What to explore in the workspace."})


# --- Sample Project Setup ---


def create_sample_project(base_dir: Path) -> None:
    """Create a simple sample project for exploration."""
    # Main source file
    src_dir = base_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)

    _ = (src_dir / "main.py").write_text("""\
\"\"\"Main entry point for the sample application.\"\"\"

from .utils import format_message
from .config import load_config


def main() -> None:
    config = load_config()
    message = format_message(config["greeting"], config["name"])
    print(message)


if __name__ == "__main__":
    main()
""")

    _ = (src_dir / "utils.py").write_text("""\
\"\"\"Utility functions.\"\"\"


def format_message(greeting: str, name: str) -> str:
    return f"{greeting}, {name}!"


def validate_name(name: str) -> bool:
    # TODO: Add proper validation
    return bool(name)
""")

    _ = (src_dir / "config.py").write_text("""\
\"\"\"Configuration management.\"\"\"

import json
from pathlib import Path

CONFIG_PATH = Path(__file__).parent.parent / "config.json"


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {"greeting": "Hello", "name": "World"}
    with open(CONFIG_PATH) as f:
        return json.load(f)
""")

    # Config file
    _ = (base_dir / "config.json").write_text("""\
{
    "greeting": "Hello",
    "name": "Developer"
}
""")

    # README
    _ = (base_dir / "README.md").write_text("""\
# Sample Project

A simple demonstration project.

## Structure

- `src/main.py` - Entry point
- `src/utils.py` - Utility functions
- `src/config.py` - Configuration loading
- `config.json` - Runtime configuration

## Usage

```bash
python -m src.main
```
""")


# --- Prompt Template Builder ---


def build_template(
    session: Session, project_dir: Path
) -> PromptTemplate[ExplorationResult]:
    """Build the prompt template with VFS and planning tools.

    The VfsToolsSection creates a sandboxed virtual filesystem with the
    project directory mounted as read-only.
    """
    return PromptTemplate[ExplorationResult](
        ns="examples/progressive",
        key="workspace-explorer",
        name="workspace_explorer",
        sections=(
            MarkdownSection[ExplorationParams](
                title="Instructions",
                template="""
You are a code exploration assistant with access to a virtual filesystem.

Your task: ${task}

Use the available tools to:
1. First create a plan for your exploration
2. List directories to understand the project structure
3. Read files to understand the code
4. Use grep to search for patterns if needed
5. Update your plan as you discover things

When done, provide a comprehensive summary of your findings.

Respond with JSON containing:
- summary: Overview of what you discovered
- files_examined: List of files you read
- key_findings: Important discoveries
- suggestions: Recommendations for the codebase
                """,
                key="instructions",
            ),
            # Optional: WorkspaceDigestSection shows a cached summary
            # (initially empty, populated by optimizers in production)
            WorkspaceDigestSection(session=session),
            # Planning tools for structured exploration
            PlanningToolsSection(
                session=session,
                strategy=PlanningStrategy.PLAN_ACT_REFLECT,
            ),
            # VFS tools for file exploration
            VfsToolsSection(
                session=session,
                mounts=(
                    HostMount(
                        host_path=str(project_dir),
                        mount_path=VfsPath(("project",)),
                        include_glob=("*.py", "*.md", "*.json", "*.txt"),
                        max_bytes=100_000,
                    ),
                ),
                allowed_host_roots=(str(project_dir.parent),),
            ),
        ),
    )


def render_plan(session: Session) -> str:
    """Render the current plan state for display."""
    plan = session[Plan].latest()
    if plan is None or not plan.objective:
        return "No plan created yet."

    lines = [f"Objective: {plan.objective}"]
    for step in plan.steps:
        marker = "[x]" if step.status == "done" else "[ ]"
        lines.append(f"  {marker} {step.step_id}. {step.title}")
    return "\n".join(lines)


def main() -> None:
    """Explore a sample project using VFS tools."""
    if "OPENAI_API_KEY" not in os.environ:
        raise SystemExit("Set OPENAI_API_KEY to run this example.")

    # Create a temporary project to explore
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir) / "sample_project"
        project_dir.mkdir()
        create_sample_project(project_dir)

        print("=" * 60)
        print("Workspace Explorer Example")
        print("=" * 60)
        print(f"\nCreated sample project at: {project_dir}")
        print("\nProject structure:")
        for path in sorted(project_dir.rglob("*")):
            if path.is_file():
                rel = path.relative_to(project_dir)
                print(f"  {rel}")

        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        adapter = OpenAIAdapter(model=model)
        session = Session()
        template = build_template(session, project_dir)

        # Give the LLM an exploration task
        task = "Explore this Python project and identify any potential issues or improvements."

        print(f"\nTask: {task}")
        print("\n" + "-" * 60)
        print("Exploring...")
        print("-" * 60)

        prompt = Prompt(template).bind(ExplorationParams(task=task))
        response = adapter.evaluate(prompt, session=session)

        if response.output is not None:
            result = response.output
            print(f"\nSummary:\n{result.summary}")

            print("\nFiles examined:")
            for f in result.files_examined:
                print(f"  - {f}")

            print("\nKey findings:")
            for finding in result.key_findings:
                print(f"  - {finding}")

            print("\nSuggestions:")
            for suggestion in result.suggestions:
                print(f"  - {suggestion}")
        else:
            print(f"\nRaw response: {response.text or '(no response)'}")

        print("\n" + "-" * 60)
        print("Plan used during exploration:")
        print("-" * 60)
        print(render_plan(session))


if __name__ == "__main__":
    main()
