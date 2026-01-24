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

"""Task example sections for trajectory-based demonstrations."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final, Self, cast, override

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


@dataclass(slots=True, frozen=True)
class TaskStep[ParamsT: SupportsDataclassOrNone, ResultT: SupportsToolResult]:
    """Single tool invocation in a task trajectory."""

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


class TaskExample[ParamsT: SupportsDataclass](Section[ParamsT]):
    """Section representing a single task trajectory example.

    The outcome type must match the PromptTemplate's output type:
    - For prompts with structured output (PromptTemplate[OutputType]), outcome must be OutputType
    - For prompts without structured output, outcome must be a string

    This class is generic over the params type (ParamsT) for use with enabled predicates.
    When no params are needed, simply use TaskExample without type arguments.
    Use TaskExample[YourParamsType] when you need parameterized enabled predicates.
    """

    # Default to no params type - users can specialize via TaskExample[ParamsType]
    _params_type = None

    outcome: str | SupportsDataclass

    def __init__(  # noqa: PLR0913
        self,
        *,
        key: str,
        objective: str,
        outcome: str | SupportsDataclass,
        steps: Sequence[TaskStep[Any, Any]],
        title: str | None = None,
        default_params: ParamsT | None = None,
        enabled: EnabledPredicate | None = None,
        accepts_overrides: bool = True,
        summary: str | None = None,
        visibility: VisibilitySelector = SectionVisibility.FULL,
    ) -> None:
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


class TaskExamplesSection(Section[SupportsDataclass]):
    """Container section for task example children.

    This class is non-generic as it serves only as a container for TaskExample
    instances. The individual TaskExample children can still be parameterized.
    """

    # Override _params_type to None since this container doesn't use params
    _params_type = None

    def __init__(  # noqa: PLR0913
        self,
        *,
        key: str = "task-examples",
        title: str = "Task Examples",
        examples: Sequence[TaskExample[Any]],
        default_params: SupportsDataclass | None = None,
        enabled: EnabledPredicate | None = None,
        accepts_overrides: bool = True,
        summary: str | None = None,
        visibility: VisibilitySelector = SectionVisibility.FULL,
    ) -> None:
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
        del path, session
        effective = self.effective_visibility(override=visibility, params=params)
        if effective == SectionVisibility.SUMMARY and self.summary is not None:
            return self.summary
        # Container renders just the heading; children are rendered by the registry
        return ""

    @override
    def clone(self, **kwargs: object) -> Self:
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
