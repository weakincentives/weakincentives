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

"""Task completion checking for prompt-scoped verification.

This module provides the protocol and built-in implementations for verifying
task completion before allowing an agent to stop. Checkers are declared on the
``PromptTemplate`` alongside policies and feedback providers.

Example:
    >>> from weakincentives.prompt import FileOutputChecker, PromptTemplate
    >>>
    >>> template = PromptTemplate(
    ...     ns="my-agent",
    ...     key="main",
    ...     sections=[...],
    ...     task_completion_checker=FileOutputChecker(
    ...         files=("report.md", "results.json"),
    ...     ),
    ... )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ..dataclasses import FrozenDataclass
from ..runtime.logging import get_logger

if TYPE_CHECKING:
    from ..filesystem import Filesystem
    from ..runtime.session import SessionProtocol

__all__ = [
    "CompositeChecker",
    "FileOutputChecker",
    "TaskCompletionChecker",
    "TaskCompletionContext",
    "TaskCompletionResult",
]

logger = get_logger(__name__)


@FrozenDataclass()
class TaskCompletionResult:
    """Result of a task completion check.

    Attributes:
        complete: True if all tasks are complete and the agent can stop.
        feedback: Natural language feedback explaining the completion status.
            When complete is False, this contains a reminder of what remains.
            When complete is True, this may contain a summary of completed work.
    """

    complete: bool
    feedback: str | None = None

    @classmethod
    def ok(cls, feedback: str | None = None) -> TaskCompletionResult:
        """Create a result indicating tasks are complete."""
        return cls(complete=True, feedback=feedback)

    @classmethod
    def incomplete(cls, feedback: str) -> TaskCompletionResult:
        """Create a result indicating tasks are not complete."""
        return cls(complete=False, feedback=feedback)


@dataclass(slots=True)
class TaskCompletionContext:
    """Context provided to task completion checkers.

    Attributes:
        session: The session containing state slices.
        tentative_output: The output the agent is attempting to produce.
        filesystem: Optional filesystem for checking file-based completion.
        adapter: Optional adapter for LLM-based verification.
        stop_reason: The reason for the stop attempt.
    """

    session: SessionProtocol
    tentative_output: Any = None
    filesystem: Filesystem | None = None
    adapter: object | None = None
    stop_reason: str | None = None


@runtime_checkable
class TaskCompletionChecker(Protocol):
    """Protocol for task completion verification.

    Implementations can check various aspects of completion:
    - File existence (required outputs created)
    - LLM verification (using an LLM to judge completion)
    - Custom business logic

    Checkers are designed to be composable via CompositeChecker.
    """

    def check(self, context: TaskCompletionContext) -> TaskCompletionResult:
        """Check if tasks are complete.

        Args:
            context: Context with session, output, filesystem, and adapter.

        Returns:
            Result indicating completion status with natural language feedback.
        """
        ...


class FileOutputChecker:
    """File-based task completion checker.

    Verifies that required output files exist on the filesystem. Accepts a
    tuple of file paths; returns incomplete if any are missing.

    Fails open when no filesystem is available in the context (cannot verify
    without filesystem access).

    Example:
        >>> checker = FileOutputChecker(files=("report.md", "results.json"))
    """

    def __init__(self, files: tuple[str, ...]) -> None:
        """Initialize the checker.

        Args:
            files: Paths that must exist for task completion.
        """
        super().__init__()
        self._files = files

    def check(self, context: TaskCompletionContext) -> TaskCompletionResult:
        """Check if all required output files exist.

        Args:
            context: Context with optional filesystem for existence checks.

        Returns:
            Complete if all files exist or no filesystem available.
            Incomplete with feedback listing missing files otherwise.
        """
        if context.filesystem is None:
            logger.debug(
                "task_completion.file_output_checker.no_filesystem",
                event="file_output_checker.no_filesystem",
            )
            return TaskCompletionResult.ok("No filesystem; cannot verify outputs.")

        missing = [f for f in self._files if not context.filesystem.exists(f)]
        if not missing:
            file_count = len(self._files)
            logger.debug(
                "task_completion.file_output_checker.all_exist",
                event="file_output_checker.all_exist",
                context={"file_count": file_count},
            )
            return TaskCompletionResult.ok(
                f"All {file_count} required output(s) exist."
            )

        missing_count = len(missing)
        max_in_message = 3
        file_list = ", ".join(missing[:max_in_message])
        if len(missing) > max_in_message:
            file_list += "..."

        logger.info(
            "task_completion.file_output_checker.missing",
            event="file_output_checker.missing",
            context={
                "missing_count": missing_count,
                "total_count": len(self._files),
                "missing_files": missing[:5],
            },
        )

        return TaskCompletionResult.incomplete(
            f"<blocker>\n{missing_count} required output file(s) not found: {file_list}\n</blocker>"
        )


class CompositeChecker:
    """Combines multiple checkers with configurable logic.

    Checkers are evaluated in order. By default (all_must_pass=True),
    all checkers must return complete for the composite to be complete.
    With all_must_pass=False, any checker returning complete is sufficient.
    """

    def __init__(
        self,
        checkers: tuple[TaskCompletionChecker, ...],
        *,
        all_must_pass: bool = True,
    ) -> None:
        """Initialize the composite checker.

        Args:
            checkers: Tuple of checkers to evaluate.
            all_must_pass: If True (default), all checkers must pass.
                If False, any checker passing is sufficient.
        """
        super().__init__()
        self._checkers = checkers
        self._all_must_pass = all_must_pass

    def check(self, context: TaskCompletionContext) -> TaskCompletionResult:
        """Check completion using all configured checkers.

        Args:
            context: Context passed to each checker.

        Returns:
            Combined result based on all_must_pass setting.
        """
        if not self._checkers:
            return TaskCompletionResult.ok("No checkers configured.")

        results: list[TaskCompletionResult] = []
        for checker in self._checkers:
            result = checker.check(context)
            results.append(result)

            # Short-circuit based on mode
            if self._all_must_pass and not result.complete:
                # First failure stops evaluation
                return result
            if not self._all_must_pass and result.complete:
                # First success stops evaluation
                return result

        # All checkers evaluated
        if self._all_must_pass:
            # All passed
            feedbacks = [r.feedback for r in results if r.feedback]
            combined = " ".join(feedbacks) if feedbacks else "All checks passed."
            return TaskCompletionResult.ok(combined)

        # None passed (only reached if all_must_pass=False)
        feedbacks = [r.feedback for r in results if r.feedback and not r.complete]
        combined = " ".join(feedbacks) if feedbacks else "No checks passed."
        return TaskCompletionResult.incomplete(combined)
