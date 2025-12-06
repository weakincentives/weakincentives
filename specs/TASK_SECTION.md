# Task Section Specification

## Overview

The `TaskSection` abstraction provides a canonical way to represent user intent in prompt
templates. It defines a `Task` dataclass that captures the user's request along with
optional background context. When progressive disclosure triggers a re-render, expansion
instructions flow through the task object itself via rebinding.

## Rationale

Agents need structured access to user intent. Ad-hoc parameter passing leads to
inconsistent handling across prompts. By standardizing on a `Task` base class we ensure:

- **Consistent intent representation** across all prompts.
- **Purposeful disclosure** where expansion context flows naturally through the task.

## Scope

- Covers the `Task` dataclass, `TaskSection`, and expansion flow.
- Documents integration with the code reviewer example.

## Task Dataclass

### Core Structure

```python
from dataclasses import dataclass, field

from weakincentives.dataclasses import FrozenDataclass


@FrozenDataclass()
class Task:
    """Base dataclass for capturing user intent.

    Subclass this to define domain-specific task types with additional fields.
    """

    request: str = field(
        metadata={"description": "The user's request in natural language."},
    )
    background: str | None = field(
        default=None,
        metadata={"description": "Additional context or background for the request."},
    )
    expansion_instructions: str | None = field(
        default=None,
        metadata={
            "description": (
                "Instructions injected after sections are expanded via progressive "
                "disclosure. Guides the model on how to proceed with newly visible content."
            ),
        },
    )
```

The `expansion_instructions` field is populated when rebinding after a
`VisibilityExpansionRequired` exception, providing continuity between render cycles.

### Domain-Specific Tasks

Prompts define subclasses that extend `Task` with domain-specific fields:

```python
@FrozenDataclass()
class CodeReviewTask(Task):
    """Task for code review requests."""

    files: tuple[str, ...] | None = field(
        default=None,
        metadata={"description": "Specific files to focus the review on."},
    )
    focus: str | None = field(
        default=None,
        metadata={"description": "Review focus such as 'security' or 'performance'."},
    )
```

## TaskSection Component

The `TaskSection` renders task content with optional background and expansion context:

```python
class TaskSection(MarkdownSection[TaskT]):
    """Section that renders a Task with optional background and expansion context."""

    def __init__(
        self,
        *,
        key: str = "task",
        title: str = "Task",
        **kwargs: object,
    ) -> None:
        template = "${_expansion_block}${request}\n\n${_background_block}"
        super().__init__(title=title, key=key, template=template, **kwargs)
```

### Rendered Output

Initial render (no expansion):

```markdown
## 5. Task

Review the authentication module for security vulnerabilities.

**Background:** Follow-up to the Q4 security audit findings.
```

After expansion with instructions:

```markdown
## 5. Task

**Expansion Context:** Sections expanded: `reference-docs`. Reason: Need security
guidelines. Continue with your task using the newly visible content.

---

Review the authentication module for security vulnerabilities.

**Background:** Follow-up to the Q4 security audit findings.
```

## VisibilityExpansionRequired Extension

The `VisibilityExpansionRequired` exception gains an `expansion_instructions` attribute:

```python
class VisibilityExpansionRequired(PromptError):
    """Raised when the model requests expansion of summarized sections."""

    requested_overrides: Mapping[tuple[str, ...], SectionVisibility]
    reason: str
    section_keys: tuple[str, ...]

    expansion_instructions: str | None = None
    """Guidance on how to proceed after sections are expanded."""
```

The `open_sections` handler constructs instructions from the model's stated reason:

```python
def build_expansion_instructions(
    section_keys: tuple[str, ...],
    reason: str,
) -> str:
    """Build guidance for continuing after expansion."""
    keys = ", ".join(f"`{k}`" for k in section_keys)
    return (
        f"Sections expanded: {keys}. "
        f"Reason: {reason}. "
        "Continue with your task using the newly visible content."
    )
```

## Evaluation Loop Pattern

The evaluation loop catches `VisibilityExpansionRequired`, updates the task with
expansion instructions using `dataclasses.replace`, and rebinds:

```python
def _evaluate_turn(self, user_prompt: str) -> str:
    task = CodeReviewTask(request=user_prompt)
    bound = self.prompt.bind(task)

    while True:
        try:
            response = self.adapter.evaluate(
                bound,
                visibility_overrides=self.visibility_overrides,
            )
            return _render_response_payload(response)
        except VisibilityExpansionRequired as e:
            self.visibility_overrides.update(e.requested_overrides)
            task = replace(task, expansion_instructions=e.expansion_instructions)
            bound = self.prompt.bind(task)
```

## Code Reviewer Integration

### Migration

Replace `ReviewTurnParams` with `CodeReviewTask`:

```python
@FrozenDataclass()
class CodeReviewTask(Task):
    """Task for code review requests."""

    files: tuple[str, ...] | None = None
    focus: str | None = None
```

Update prompt construction:

```python
sections = (
    _build_review_guidance_section(),
    WorkspaceDigestSection(session=session),
    _build_reference_section(),
    PlanningToolsSection(session=session, strategy=PlanningStrategy.PLAN_ACT_REFLECT),
    _build_workspace_section(session=session),
    TaskSection[CodeReviewTask](title="Review Task", key="review-task"),
)
```

Update the evaluation loop to rebind with expansion instructions as shown above.

## Implementation Checklist

- [ ] `Task` base dataclass with `expansion_instructions` field
- [ ] `TaskSection` component with expansion block rendering
- [ ] Add `expansion_instructions` to `VisibilityExpansionRequired`
- [ ] Update `open_sections` handler to populate instructions
- [ ] Migrate code reviewer example

## Testing Requirements

- Unit tests for `Task` field defaults including `expansion_instructions`
- Unit tests for `TaskSection` rendering with and without expansion context
- Unit tests for `expansion_instructions` in `VisibilityExpansionRequired`
- Integration tests for replace + bind flow after expansion
- Code reviewer example tests

## Caveats

- **Fields are optional**: Prompts render cleanly when background or expansion
  instructions are absent.
- **Expansion instructions are transient per-cycle**: Updated on each rebind after
  `VisibilityExpansionRequired`, cleared on fresh task construction.
- **No nested tasks**: Delegation uses the existing `DelegationPrompt` mechanism.
