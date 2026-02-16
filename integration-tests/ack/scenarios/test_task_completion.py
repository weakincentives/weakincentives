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

"""Tier 4 ACK scenarios for task completion checking.

Task completion checkers are declared on the prompt and enforced by the
adapter to prevent premature stopping. These scenarios verify that adapters
block early stops when required output files are missing and allow stops
when all files exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from weakincentives.adapters.core import ProviderAdapter
from weakincentives.filesystem import Filesystem, HostFilesystem
from weakincentives.prompt import (
    MarkdownSection,
    Prompt,
    PromptTemplate,
    Tool,
    ToolContext,
    ToolResult,
)
from weakincentives.prompt.task_completion import FileOutputChecker
from weakincentives.runtime.session import Session

from ..adapters import AdapterFixture
from . import make_adapter_ns

pytestmark = pytest.mark.ack_capability("task_completion")


@dataclass(slots=True)
class WriteFileParams:
    """Input for the write-file tool."""

    path: str
    content: str


@dataclass(slots=True, frozen=True)
class WriteFileResult:
    """Output from the write-file tool."""

    written: bool
    path: str


def _build_workspace_write_tool(
    workspace: Path,
) -> Tool[WriteFileParams, WriteFileResult]:
    """Build a tool that writes to the workspace directory."""

    def handler(
        params: WriteFileParams,
        *,
        context: ToolContext,
    ) -> ToolResult[WriteFileResult]:
        del context
        target = workspace / params.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(params.content)
        result = WriteFileResult(written=True, path=params.path)
        return ToolResult.ok(result, message=f"Wrote {params.path}")

    return Tool[WriteFileParams, WriteFileResult](
        name="write_workspace_file",
        description="Write content to a file in the workspace.",
        handler=handler,
    )


@dataclass(slots=True)
class TaskParams:
    """Prompt params for the task completion scenarios."""

    output_file: str


def test_task_completion_blocks_early_stop(
    adapter_fixture: AdapterFixture,
    session: Session,
    tmp_path: Path,
) -> None:
    """Adapter blocks stop when required output files are missing.

    The prompt configures a FileOutputChecker requiring ``output.txt``.
    The agent is instructed to create the file using the provided tool.
    We verify that the adapter does not stop until the file exists,
    demonstrated by the file being present after evaluation.
    """
    adapter = adapter_fixture.create_adapter(tmp_path)
    tool = _build_workspace_write_tool(tmp_path)
    output_file = "output.txt"

    section = MarkdownSection[TaskParams](
        title="File Task",
        template=(
            "Create a file named '${output_file}' using the "  # noqa: RUF027
            "`write_workspace_file` tool. Write some brief content to it. "
            "Then respond with 'done'."
        ),
        tools=(tool,),
        key="task",
    )

    template = PromptTemplate(
        ns=make_adapter_ns(adapter_fixture.adapter_name),
        key="ack-task-completion",
        name="ack_task_completion",
        sections=[section],
        task_completion_checker=FileOutputChecker(
            files=(str(tmp_path / output_file),),
        ),
    )

    fs = HostFilesystem(_root=str(tmp_path))
    prompt = Prompt(template).bind(
        TaskParams(output_file=output_file),
        resources={Filesystem: fs},
    )
    response = adapter.evaluate(prompt, session=session)

    # The agent should have created the file
    assert (tmp_path / output_file).exists()
    assert response.text is not None


def test_task_completion_allows_stop_when_complete(
    adapter_fixture: AdapterFixture,
    session: Session,
    tmp_path: Path,
) -> None:
    """Adapter allows stop when all required output files already exist.

    Pre-create the required file. The agent should be allowed to stop
    immediately after producing its response.
    """
    adapter = adapter_fixture.create_adapter(tmp_path)
    output_file = "output.txt"

    # Pre-create the required file
    (tmp_path / output_file).write_text("pre-existing content")

    section = MarkdownSection(
        title="Pre-completed Task",
        template=(
            "The required output file already exists. "
            "Reply with 'already done' to confirm."
        ),
        key="task",
    )

    template = PromptTemplate(
        ns=make_adapter_ns(adapter_fixture.adapter_name),
        key="ack-task-complete",
        name="ack_task_complete",
        sections=[section],
        task_completion_checker=FileOutputChecker(
            files=(str(tmp_path / output_file),),
        ),
    )

    fs = HostFilesystem(_root=str(tmp_path))
    prompt = Prompt(template).bind(resources={Filesystem: fs})
    response = adapter.evaluate(prompt, session=session)

    # Agent should have responded without being blocked
    assert response.text is not None


def test_no_checker_allows_free_stop(
    adapter: ProviderAdapter[object],
    session: Session,
    adapter_fixture: AdapterFixture,
) -> None:
    """Adapter allows free stop when no task completion checker is configured."""
    section = MarkdownSection(
        title="Simple Task",
        template="Reply with 'hello' in one word.",
        key="task",
    )

    template = PromptTemplate(
        ns=make_adapter_ns(adapter_fixture.adapter_name),
        key="ack-no-checker",
        name="ack_no_checker",
        sections=[section],
        # No task_completion_checker
    )

    prompt = Prompt(template)
    response = adapter.evaluate(prompt, session=session)

    assert response.text is not None
