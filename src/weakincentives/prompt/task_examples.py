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

"""Task example sections for trajectory-based demonstrations.

This module provides section types for embedding task trajectory examples
into prompts. Task examples demonstrate how to accomplish specific objectives
through sequences of tool invocations, helping LLMs understand expected
tool usage patterns.

Classes:
    TaskStep: A single tool invocation step within a trajectory.
    TaskExample: A section containing a complete task trajectory.
    TaskExamplesSection: A container grouping multiple TaskExample children.

Example:
    Create a task example showing file search workflow::

        from weakincentives.prompt import TaskExample, TaskStep, ToolExample

        example = TaskExample(
            key="search-example",
            objective="Find all configuration files",
            steps=[
                TaskStep(
                    tool_name="glob_files",
                    example=ToolExample(
                        description="Search for config files",
                        input=GlobParams(pattern="*.yaml"),
                        output=GlobResult(files=["config.yaml"]),
                    ),
                ),
            ],
            outcome="Found 1 configuration file",
        )
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final, Self, TypeVar, cast, override

from ..serde import clone as clone_dataclass, dump
from ..types.dataclass import (
    SupportsDataclass,
    SupportsDataclassOrNone,
    SupportsToolResult,
)
from ._enabled_predicate import EnabledPredicate
from ._visibility import SectionVisibility, VisibilitySelector
from .errors import PromptValidationError
from .section import Section
from .tool import ToolExample

_OBJECTIVE_MIN_LENGTH: Final = 1
_OBJECTIVE_MAX_LENGTH: Final = 500
_TITLE_TRUNCATE_LENGTH: Final = 60

_TOOL_NAME_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9_-]{1,64}$")

TaskExampleParamsT = TypeVar(
    "TaskExampleParamsT", bound=SupportsDataclass, covariant=True
)
TaskExamplesParamsT = TypeVar(
    "TaskExamplesParamsT", bound=SupportsDataclass, covariant=True
)
OutcomeT = TypeVar("OutcomeT", bound="str | SupportsDataclass", covariant=True)


@dataclass(slots=True, frozen=True)
class TaskStep[ParamsT: SupportsDataclassOrNone, ResultT: SupportsToolResult]:
    """A single tool invocation step within a task trajectory.

    TaskStep pairs a tool name with a ToolExample demonstrating its usage.
    Multiple TaskSteps form the trajectory showing how to accomplish a task.

    Attributes:
        tool_name: Name of the tool to invoke. Must match pattern ``^[a-z0-9_-]{1,64}$``.
        example: A ToolExample containing the input parameters and expected output
            for this tool invocation.

    Example:
        >>> step = TaskStep(
        ...     tool_name="search_files",
        ...     example=ToolExample(
        ...         description="Search for Python files",
        ...         input=SearchParams(pattern="*.py"),
        ...         output=SearchResult(files=["main.py"]),
        ...     ),
        ... )
    """

    tool_name: str
    example: ToolExample[ParamsT, ResultT]


def _validate_ascii_text(
    value: str,
    *,
    field_name: str,
    min_length: int,
    max_length: int,
    section_path: tuple[str, ...],
) -> str:
    """Validate ASCII text field with length constraints."""
    stripped = value.strip()
    if not stripped or len(stripped) < min_length:
        raise PromptValidationError(
            f"{field_name} must not be empty",
            section_path=section_path,
            placeholder=field_name.lower(),
        )
    if len(stripped) > max_length:
        raise PromptValidationError(
            f"{field_name} must be <= {max_length} characters",
            section_path=section_path,
            placeholder=field_name.lower(),
        )
    try:
        _ = stripped.encode("ascii")
    except UnicodeEncodeError as error:
        raise PromptValidationError(
            f"{field_name} must contain only printable ASCII characters",
            section_path=section_path,
            placeholder=field_name.lower(),
        ) from error
    return stripped


def _validate_tool_name(tool_name: str, *, section_path: tuple[str, ...]) -> str:
    """Validate tool name format."""
    stripped = tool_name.strip()
    if not _TOOL_NAME_PATTERN.match(stripped):
        raise PromptValidationError(
            f"Invalid tool name format: {tool_name!r}",
            section_path=section_path,
            placeholder="tool_name",
        )
    return stripped


def _truncate_title(objective: str, max_length: int = _TITLE_TRUNCATE_LENGTH) -> str:
    """Truncate objective to create a title."""
    if len(objective) <= max_length:
        return objective
    return objective[: max_length - 3] + "..."


def _render_step_value(value: SupportsDataclass | None) -> str:
    """Serialize step input/output to JSON string."""
    if value is None:
        return "null"
    serialized = dump(value, exclude_none=True)
    return json.dumps(serialized, ensure_ascii=False)


def _render_outcome(outcome: str | SupportsDataclass) -> str:
    """Render outcome as string or JSON for dataclass types."""
    if isinstance(outcome, str):
        return outcome
    serialized = dump(outcome, exclude_none=True)
    return json.dumps(serialized, ensure_ascii=False)


class TaskExample(Section[TaskExampleParamsT]):
    """A section representing a complete task trajectory example.

    TaskExample demonstrates how to accomplish a specific objective through
    a sequence of tool invocations. It renders as structured markdown showing
    the objective, each step with its tool inputs/outputs, and the final outcome.

    The outcome type must match the PromptTemplate's output type:
    - For prompts with structured output (PromptTemplate[OutputType]), outcome
      must be an instance of OutputType
    - For prompts without structured output, outcome must be a string

    Attributes:
        objective: The task objective (1-500 ASCII characters).
        outcome: The final result after completing all steps.
        steps: Tuple of TaskStep instances forming the trajectory.

    Example:
        >>> example = TaskExample(
        ...     key="file-search-example",
        ...     objective="Find and list all Python test files",
        ...     steps=[
        ...         TaskStep(
        ...             tool_name="glob_files",
        ...             example=ToolExample(
        ...                 description="Search for test files",
        ...                 input=GlobParams(pattern="test_*.py"),
        ...                 output=GlobResult(files=["test_main.py"]),
        ...             ),
        ...         ),
        ...     ],
        ...     outcome="Found 1 test file: test_main.py",
        ... )
    """

    outcome: str | SupportsDataclass

    def __init__(  # noqa: PLR0913
        self,
        *,
        key: str,
        objective: str,
        outcome: str | SupportsDataclass,
        steps: Sequence[TaskStep[Any, Any]],
        title: str | None = None,
        default_params: TaskExampleParamsT | None = None,
        enabled: EnabledPredicate | None = None,
        accepts_overrides: bool = True,
        summary: str | None = None,
        visibility: VisibilitySelector = SectionVisibility.FULL,
    ) -> None:
        """Initialize a TaskExample section.

        Args:
            key: Unique identifier for this section.
            objective: Description of the task goal (1-500 ASCII characters).
            outcome: The expected result after completing all steps. Must match
                the PromptTemplate's output type.
            steps: Non-empty sequence of TaskStep instances defining the trajectory.
            title: Display title for the section. Defaults to a truncated objective
                (max 60 characters).
            default_params: Default parameters for this section.
            enabled: Predicate controlling whether this section is enabled.
            accepts_overrides: Whether parameter overrides are allowed.
            summary: Short summary shown when visibility is SUMMARY.
            visibility: Controls section visibility (FULL, SUMMARY, or HIDDEN).

        Raises:
            PromptValidationError: If objective is empty, too long, or contains
                non-ASCII characters; if steps is empty or contains non-TaskStep
                items; or if any step has an invalid tool name.
        """
        # Validate objective
        validated_objective = _validate_ascii_text(
            objective,
            field_name="objective",
            min_length=_OBJECTIVE_MIN_LENGTH,
            max_length=_OBJECTIVE_MAX_LENGTH,
            section_path=(key,),
        )

        # Validate steps
        if not steps:
            raise PromptValidationError(
                "steps must not be empty",
                section_path=(key,),
                placeholder="steps",
            )

        validated_steps: list[TaskStep[Any, Any]] = []
        for idx, step in enumerate(steps):
            # Runtime type check - cast to object to allow isinstance
            step_obj = cast(object, step)
            if not isinstance(step_obj, TaskStep):
                raise PromptValidationError(
                    f"Step {idx} must be a TaskStep instance",
                    section_path=(key,),
                    placeholder="steps",
                )
            # Validate tool name format
            _ = _validate_tool_name(step.tool_name, section_path=(key,))
            validated_steps.append(step)

        self.objective = validated_objective
        self.outcome = outcome
        self.steps: tuple[TaskStep[Any, Any], ...] = tuple(validated_steps)

        # Derive title from objective if not provided
        display_title = (
            title if title is not None else _truncate_title(validated_objective)
        )

        super().__init__(
            title=display_title,
            key=key,
            default_params=default_params,
            children=None,
            enabled=enabled,
            tools=None,
            accepts_overrides=accepts_overrides,
            summary=summary,
            visibility=visibility,
        )

    @override
    def render_body(
        self,
        params: SupportsDataclass | None,
        *,
        visibility: SectionVisibility | None = None,
        path: tuple[str, ...] = (),
        session: object = None,
    ) -> str:
        """Render the task example body as markdown.

        Produces structured markdown with the objective, numbered steps showing
        tool invocations with JSON-formatted inputs/outputs, and the final outcome.

        Args:
            params: Section parameters (unused for rendering, but checked for
                visibility determination).
            visibility: Override for section visibility.
            path: Section path (unused).
            session: Session context (unused).

        Returns:
            Markdown string. Returns the summary if visibility is SUMMARY and
            a summary is defined; otherwise returns the full rendered body.
        """
        del path, session
        effective = self.effective_visibility(override=visibility, params=params)
        if effective == SectionVisibility.SUMMARY and self.summary is not None:
            return self.summary
        return self._render_full_body()

    def _render_full_body(self) -> str:
        """Render the full task example body without heading."""
        lines: list[str] = []
        lines.append(f"**Objective:** {self.objective}")
        lines.append("")
        lines.append("**Steps:**")
        lines.append("")

        for idx, step in enumerate(self.steps, start=1):
            description = step.example.description
            lines.append(f"{idx}. **{step.tool_name}** - {description}")

            # Render input
            input_json = _render_step_value(step.example.input)
            lines.append("   - input:")
            lines.append("     ```json")
            lines.extend(f"     {line}" for line in input_json.splitlines())
            lines.append("     ```")

            # Render output
            output_json = _render_step_value(step.example.output)
            lines.append("   - output:")
            lines.append("     ```")
            lines.extend(f"     {line}" for line in output_json.splitlines())
            lines.append("     ```")
            lines.append("")

        lines.append(f"**Outcome:** {_render_outcome(self.outcome)}")

        return "\n".join(lines)

    @override
    def clone(self, **kwargs: object) -> Self:
        """Create a deep copy of this TaskExample.

        Creates a new TaskExample instance with cloned default_params.
        The steps tuple is shared (immutable) but all other mutable state
        is copied.

        Args:
            **kwargs: Unused keyword arguments (accepted for API compatibility).

        Returns:
            A new TaskExample instance with the same configuration.
        """
        cloned_default = (
            clone_dataclass(self.default_params)
            if self.default_params is not None
            else None
        )

        cls: type[Any] = type(self)
        clone = cls(
            key=self.key,
            objective=self.objective,
            outcome=self.outcome,
            steps=self.steps,
            title=self.title,
            default_params=cloned_default,
            enabled=self._enabled,  # ty: ignore[invalid-argument-type]  # callback arity
            accepts_overrides=self.accepts_overrides,
            summary=self.summary,
            visibility=self.visibility,
        )
        return cast(Self, clone)


class TaskExamplesSection(Section[TaskExamplesParamsT]):
    """Container section that groups multiple TaskExample children.

    TaskExamplesSection organizes related task examples under a common heading.
    It renders only the heading; the individual TaskExample children are rendered
    separately by the section registry.

    Use this to group task trajectory demonstrations that illustrate how to
    accomplish various objectives using the available tools.

    Example:
        >>> section = TaskExamplesSection(
        ...     key="examples",
        ...     title="Example Tasks",
        ...     examples=[
        ...         TaskExample(
        ...             key="example-1",
        ...             objective="Find files",
        ...             steps=[...],
        ...             outcome="Found 5 files",
        ...         ),
        ...         TaskExample(
        ...             key="example-2",
        ...             objective="Edit config",
        ...             steps=[...],
        ...             outcome="Config updated",
        ...         ),
        ...     ],
        ... )
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        key: str = "task-examples",
        title: str = "Task Examples",
        examples: Sequence[TaskExample[Any]],
        default_params: TaskExamplesParamsT | None = None,
        enabled: EnabledPredicate | None = None,
        accepts_overrides: bool = True,
        summary: str | None = None,
        visibility: VisibilitySelector = SectionVisibility.FULL,
    ) -> None:
        """Initialize a TaskExamplesSection.

        Args:
            key: Unique identifier for this section. Defaults to "task-examples".
            title: Display title for the section. Defaults to "Task Examples".
            examples: Non-empty sequence of TaskExample instances to include.
            default_params: Default parameters for this section.
            enabled: Predicate controlling whether this section is enabled.
            accepts_overrides: Whether parameter overrides are allowed.
            summary: Short summary shown when visibility is SUMMARY.
            visibility: Controls section visibility (FULL, SUMMARY, or HIDDEN).

        Raises:
            PromptValidationError: If examples is empty or contains non-TaskExample
                items.
        """
        # Validate examples
        if not examples:
            raise PromptValidationError(
                "TaskExamplesSection requires at least one example",
                section_path=(key,),
                placeholder="examples",
            )

        validated_examples: list[TaskExample[Any]] = []
        for idx, example in enumerate(examples):
            # Runtime type check - cast to object to allow isinstance
            example_obj = cast(object, example)
            if not isinstance(example_obj, TaskExample):
                msg = (
                    f"TaskExamplesSection examples must be TaskExample instances. "
                    f"Got: {type(example).__name__} at index {idx}."
                )
                raise PromptValidationError(
                    msg,
                    section_path=(key,),
                    placeholder="examples",
                )
            validated_examples.append(example)

        super().__init__(
            title=title,
            key=key,
            default_params=default_params,
            children=validated_examples,
            enabled=enabled,
            tools=None,
            accepts_overrides=accepts_overrides,
            summary=summary,
            visibility=visibility,
        )

    @override
    def render_body(
        self,
        params: SupportsDataclass | None,
        *,
        visibility: SectionVisibility | None = None,
        path: tuple[str, ...] = (),
        session: object = None,
    ) -> str:
        """Render the section body.

        As a container section, this renders only the heading. Individual
        TaskExample children are rendered separately by the section registry.

        Args:
            params: Section parameters (used for visibility determination).
            visibility: Override for section visibility.
            path: Section path (unused).
            session: Session context (unused).

        Returns:
            Empty string for FULL visibility (children render separately),
            or the summary string if visibility is SUMMARY and summary is defined.
        """
        del path, session
        effective = self.effective_visibility(override=visibility, params=params)
        if effective == SectionVisibility.SUMMARY and self.summary is not None:
            return self.summary
        # Container renders just the heading; children are rendered by the registry
        return ""

    @override
    def clone(self, **kwargs: object) -> Self:
        """Create a deep copy of this TaskExamplesSection.

        Creates a new TaskExamplesSection with cloned children and default_params.
        Each TaskExample child is recursively cloned.

        Args:
            **kwargs: Keyword arguments passed to each child's clone method.

        Returns:
            A new TaskExamplesSection instance with the same configuration
            and deeply cloned children.

        Raises:
            TypeError: If any child is not a TaskExample instance.
        """
        cloned_children: list[TaskExample[Any]] = []
        for child in self.children:
            if not isinstance(child, TaskExample):  # pragma: no cover
                raise TypeError(
                    "TaskExamplesSection children must be TaskExample instances."
                )
            cloned_children.append(child.clone(**kwargs))

        cloned_default = (
            clone_dataclass(self.default_params)
            if self.default_params is not None
            else None
        )

        cls: type[Any] = type(self)
        clone = cls(
            key=self.key,
            title=self.title,
            examples=cloned_children,
            default_params=cloned_default,
            enabled=self._enabled,  # ty: ignore[invalid-argument-type]  # callback arity
            accepts_overrides=self.accepts_overrides,
            summary=self.summary,
            visibility=self.visibility,
        )
        return cast(Self, clone)


__all__ = ["TaskExample", "TaskExamplesSection", "TaskStep"]
